# This script is based on https://github.com/salesforce/LAVIS/blob/main/lavis/runners/runner_base.py

import os
import json
import time
import datetime
from pathlib import Path
import logging
import shutil  # 파일 상단에 import 추가 필요

import torch
import torch.distributed as dist
import numpy as np
from tensorboardX import SummaryWriter
from accelerate import Accelerator
import wandb

from logger import MetricLogger, SmoothedValue
from utils import get_accelerator_dataloader, setup_accelerate_logger, get_dataloader
from optims import get_optimizer, LinearWarmupCosineLRScheduler
from models.utils import setup_quantized_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class AccelerateRunner:
    def __init__(self, cfg, model, datasets, job_id, dryrun):
        self.config = cfg
        self.dryrun = dryrun
        self.lora = None
        # log
        self.output_dir = Path(self.config.config.run.output_dir) / job_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_writter = SummaryWriter(self.output_dir)

        # settings
        self.start_epoch = 0
        self.max_epoch = self.config.config.run.optims.max_epoch
        self.evaluate_only = self.config.config.run.evaluate

        # test prompt
        self.prompt_template = self.config.config.model.get("prompt_template", "")
        test_prompt_path = self.config.config.model.get("test_prompt_path", "")
        if test_prompt_path:
            try:
                with open(test_prompt_path, "r") as f:
                    self.test_prompt_dict = json.load(f)
            except:
                print("Failed to load test prompt! Try to use utf-8 encoding.")
                with open(test_prompt_path, "r", encoding="utf-8") as f:
                    self.test_prompt_dict = json.load(f)
            for k in self.test_prompt_dict.keys():
                self.test_prompt_dict[k] = self.prompt_template.format(self.test_prompt_dict[k])
        else:
            self.test_prompt_dict = None

        # model
        self._model = model
        
        # dataloaders
        self.train_loader = get_accelerator_dataloader(datasets["train"], self.config.config.run, is_train=True)
        self.valid_loader = get_accelerator_dataloader(datasets["valid"], self.config.config.run, is_train=False)
        self.test_loader = get_accelerator_dataloader(datasets["test"], self.config.config.run, is_train=False)

        # optimizer & scheduler
        self.iters_per_epoch = len(self.train_loader) if self.config.config.run.epoch_based else self.config.config.run.iters_per_epoch
        self.optimizer = get_optimizer(self._model, self.config.config.run.optims)
        self.scheduler = LinearWarmupCosineLRScheduler(
            self.optimizer,
            max_epoch=self.max_epoch,
            iters_per_epoch=self.iters_per_epoch,
            min_lr=self.config.config.run.optims.min_lr,
            init_lr=self.config.config.run.optims.init_lr,
            warmup_steps=self.config.config.run.optims.warmup_steps,
            warmup_start_lr=self.config.config.run.optims.get("warmup_start_lr", -1),
        )

        # Accelerate setup
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.config.run.accum_grad_iters
        )  # config file의 설정을 자동으로 사용
        self.model, self.optimizer, self.train_loader, self.valid_loader, self.test_loader, self.scheduler = self.accelerator.prepare(
            self._model, self.optimizer, self.train_loader, self.valid_loader, self.test_loader, self.scheduler
        )
        
        setup_accelerate_logger(self.accelerator, self.config.config.run.log_level_warning)
        
        if self.accelerator.is_local_main_process:
            wandb.login()
            wandb.init(project="audio_lm", name=cfg.config.run.exp_name)
    
        self.device = self.accelerator.device
        self.cuda_enabled = self.accelerator.device.type == "cuda"
        
        self.log_config()
        
    def unwrap_model(self, model):
        return self.accelerator.unwrap_model(model)

    def train_epoch(self, epoch):
        self.model.train()
        
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.set_accelerator(self.accelerator)
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, self.iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)

        torch.cuda.empty_cache()
        for i in metric_logger.log_every(range(self.iters_per_epoch), self.config.config.run.log_freq, header=header, logger=self.log_writter, start_step=epoch*self.iters_per_epoch):
            if i >= self.iters_per_epoch:
                break
            
            samples = next(self.train_loader)

            if not self.dryrun:
                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        loss = self.model(samples)["loss"]
                    
                    self.accelerator.backward(loss)
                    # 옵티마이저 스텝 전 스케일러 동기화
                    if self.accelerator.scaler:
                        self.accelerator.scaler.unscale_(self.optimizer)
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)  # 그래디언트 클리핑 추가
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)  # 메모리 효율적
                    self.scheduler.step(cur_epoch=epoch, cur_step=i)

                metric_logger.update(loss=loss.item())
                metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
                
                if self.accelerator.is_local_main_process:
                    wandb.log({
                        "train/iteration": i, 
                        "train/loss": loss.item(), 
                        "train/lr": self.optimizer.param_groups[0]["lr"]
                    })
            else:
                metric_logger.update(loss=0.0)
                metric_logger.update(lr=0.0)
                if self.accelerator.is_local_main_process:
                    wandb.log({"train/iteration": i, "train/loss": 0.0, "train/lr": 0.0})

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @torch.no_grad()
    def valid_epoch(self, epoch, split, decode=False, save_json=False):
        if not self.dryrun:
            model = self.unwrap_model(self.model)
            model.eval()

        dataloader = getattr(self, split + "_loader", None)
        assert dataloader is not None, "{}_loader does not exist.".format(split)

        metric_logger = MetricLogger(delimiter="  ")
        header = "Eval: data epoch: [{}]".format(epoch)

        results = []
        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0
        total_tokens = 0

        for i, samples in enumerate(metric_logger.log_every(dataloader, self.config.config.run.log_freq, header=header)):
            if not self.dryrun:
                with self.accelerator.autocast(self.model):
                    forward_result = model(samples, verbose=True)
                    
                loss = forward_result.get("loss", 0)
                correct = forward_result.get("correct", 0)
                total = forward_result.get("total", 1)
                
                # 일반 숫자로 누적
                total_loss += loss.item() * len(samples["id"])
                total_correct += correct.item()  # tensor -> float
                total_samples += len(samples["id"])
                total_tokens += total  # tensor -> float

                if self.accelerator.is_local_main_process:
                    wandb.log({
                        f"{split}/iteration": i,
                        f"{split}/loss": loss.item(),
                        f"{split}/acc": (correct / total).item()
                    })

                res = {
                    "id": samples["id"],
                    "ground_truth": samples["text"],
                    "loss": loss.item(),
                    "acc": (correct / total).item(),
                    "total": total,
                }

                if decode:
                    if model.prompt_dict:
                        if self.test_prompt_dict is None:
                            prompts = None
                        else:
                            prompts = [self.test_prompt_dict[s] for s in samples["task"]]
                            if "Q" in samples:
                                prompts = [p.format(q) if "{}" in p else p for p, q in zip(prompts, samples["Q"])]
                    else:
                        prompts = None

                    with self.accelerator.autocast():
                        text = model.generate(samples, self.config.config.run, prompts=prompts)
                    
                    res.update({
                        "text": text,
                        "prompt": prompts,
                        "task": samples["task"]
                    })

                results.append(res)

        # 모든 GPU에서의 결과를 수집 (여기서 한 번만 CUDA 텐서로 변환)
        if self.accelerator.use_distributed:
            all_loss = self.accelerator.gather(torch.tensor(total_loss, device=self.device)).sum()
            all_correct = self.accelerator.gather(torch.tensor(total_correct, device=self.device)).sum()
            all_samples = self.accelerator.gather(torch.tensor(total_samples, device=self.device)).sum()
            all_tokens = self.accelerator.gather(torch.tensor(total_tokens, device=self.device)).sum()
        else:
            all_loss = torch.tensor(total_loss, device=self.device)
            all_correct = torch.tensor(total_correct, device=self.device)
            all_samples = torch.tensor(total_samples, device=self.device)
            all_tokens = torch.tensor(total_tokens, device=self.device)

        if save_json and self.accelerator.is_local_main_process:
            self.save_result(results, self.output_dir, f"eval_{split}_epoch_{epoch}")

        ret = {
            "loss": (all_loss / all_samples).item(),
            "agg_metrics": (all_correct / all_tokens).item()
        }

        return ret

    def save_result(self, result, result_dir, filename):
        # 각 프로세스의 결과 저장
        result_file = os.path.join(result_dir, f"{filename}_rank{self.accelerator.process_index}.json")
        try:
            json.dump(result, open(result_file, "w"), ensure_ascii=False)
        except Exception as e:
            json.dump(result, open(result_file, "w", encoding="utf-8"), ensure_ascii=False)
        
        self.accelerator.wait_for_everyone()
        
        # 메인 프로세스에서 결과 병합
        if self.accelerator.is_local_main_process:
            merged_result = []
            for rank in range(self.accelerator.num_processes):
                rank_file = os.path.join(result_dir, f"{filename}_rank{rank}.json")
                with open(rank_file, "r", encoding="utf-8") as f:
                    merged_result.extend(json.load(f))
                
            # 최종 결과 저장
            final_result_file = os.path.join(result_dir, f"{filename}.json")
            json.dump(merged_result, open(final_result_file, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"Result file saved to {final_result_file}")
            
            # 임시 파일 삭제
            for rank in range(self.accelerator.num_processes):
                os.remove(os.path.join(result_dir, f"{filename}_rank{rank}.json"))

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_epoch = 0

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            if self.evaluate_only:
                break

            # training phase
            logging.info("Training Phase")
            train_stats = self.train_epoch(cur_epoch)
            self.log_stats(train_stats, split_name="train")

            # validating phase
            logging.info("Validating Phase")
            valid_log = self.valid_epoch(cur_epoch, "valid", decode=False, save_json=False)
            if valid_log is not None:
                if self.accelerator.is_main_process:
                    agg_metrics = valid_log["agg_metrics"]
                    if agg_metrics > best_agg_metric:
                        best_agg_metric = agg_metrics
                        best_epoch = cur_epoch
                        self.save_checkpoint(cur_epoch, is_best=True)

                    valid_log.update({"best_epoch": best_epoch})
                    self.log_stats(valid_log, split_name="valid")
                    wandb.log({"valid/epoch": cur_epoch, "valid/agg_metrics": agg_metrics})

            self.save_checkpoint(cur_epoch, is_best=False)

            if self.accelerator.use_distributed:
                dist.barrier()

        # testing phase
        if self.evaluate_only:
            test_log = self.valid_epoch("best", "test", decode=True, save_json=True)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))
        
        self.save_quantized_llm_model(self.model, self.output_dir)

    def log_config(self):
        if self.accelerator.is_local_main_process:
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")

    def log_stats(self, stats, split_name):
        if self.accelerator.is_local_main_process:
            if isinstance(stats, dict):
                log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
                with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            elif isinstance(stats, list):
                pass

    def save_checkpoint(self, cur_epoch, is_best=False):
        if not self.accelerator.is_local_main_process:
            return
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = self.unwrap_model(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.accelerator.scaler.state_dict() if self.accelerator.scaler else None,
            "epoch": cur_epoch,
        }
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to) 
        
        if is_best:
            self.lora = model_no_ddp.llama_model.unload_adapter()
        
    def save_quantized_llm_model(self, output_dir):
        merged_dir = os.path.join(output_dir, "llm_lora")

        self.model.merge_lora_and_save(self.lora, merged_dir)
        
        # 양자화된 모델 저장
        quantize_dir = os.path.join(output_dir, "quantized_model")
        quantized_model = setup_quantized_model(merged_dir, self.token, self.config.config.model.ptq_method, is_train=False)
        if self.config.config.model.ptq_method == "awq":
            quantized_model.save_quantized(quantize_dir)
        else:   
            quantized_model.save_pretrained(quantize_dir)
        
        # 병합된 모델 디렉토리 삭제
        if os.path.exists(merged_dir):
            shutil.rmtree(merged_dir)
            logging.info(f"Removed temporary merged model directory: {merged_dir}")
        
                