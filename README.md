# 📖 Overview
![](https://i.imgur.com/SqupAoR.png)
Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로, 특히, 딥러닝 기술을 이용한 뼈 Segmentation은 많은 연구가 이루어지고 있으며, 다양한 목적으로 도움을 줄 수 있습니다.

<br>

## 🗂 Dataset
- **Input :** hand bone x-ray 객체가 담긴 이미지가 모델의 인풋으로 사용됩니다. segmentation annotation은 json file로 제공됩니다.
- **Output :** 모델은 각 픽셀 좌표에 따른 class를 출력하고, 이를 rle로 변환하여 리턴합니다. 이를 output 양식에 맞게 csv 파일을 만들어 제출합니다.
<br><br>
- 전체 이미지 개수: 800장(Train), 288장(Test)
- 크게 손가락 / 손등 / 팔로 구성되며, 총 29개의 class (뼈 종류)가 존재합니다.
<br><br><br>

## 📃 Metric
![image](https://github.com/user-attachments/assets/f77da0ea-caf8-4e15-a592-dab7f6c331b0)
2 * (예측 영역 ∩ 실제 영역) / (예측 영역의 크기 + 실제 영역의 크기)인 DICE score는 예측된 영역과 실제 영역 간의 중첩 정도를 수치화하여 표현합니다. 
이 공식은 두 영역이 완전히 일치할 때 최대값인 1을 갖고, 전혀 겹치지 않을 때 최소값인 0을 갖습니다.

<br><br>
# Team CV-13

## 🧑‍💻 Members 
<table>
    <tr height="160px">
        <td align="center" width="150px">
            <a href="https://github.com/boyamie"><img height="80px"  src="https://github.com/user-attachments/assets/adeaf63c-a763-46df-bd49-1a0ce71098eb"></a>
            <br/>
            <a href="https://github.com/boyamie"><strong>김보현</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/Ja2Hw"><img height="80px"  src="https://github.com/user-attachments/assets/d824f102-e0a5-491d-9c75-cb90f625da3e"/></a>
            <br/>
            <a href="https://github.com/Ja2Hw"><strong>김재환</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/Jin-SukKim"><img height="80px"  src="https://github.com/user-attachments/assets/f15196cd-96fa-404c-b418-dc84e5ced92a"/></a>
            <br/>
            <a href="https://github.com/Jin-SukKim"><strong>김진석</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/202250274"><img height="80px" src="https://github.com/user-attachments/assets/534a7596-2c95-4b89-867d-839a7728303c"/></a>
            <br />
            <a href="https://github.com/202250274"><strong>박진영</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/Superl3"><img height="80px" src="https://github.com/user-attachments/assets/3673ecc7-399b-42b0-9d94-cfcfd32d3864"/></a>
            <br />
            <a href="https://github.com/Superl3"><strong>성기훈</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/hocheol0303"><img height="80px"  src="https://github.com/user-attachments/assets/2d0a71c6-9752-43a8-b96e-bc3be06e5dde"/></a>
              <br />
              <a href="https://github.com/hocheol0303"><strong>양호철</strong></a>
              <br />
          </td>
    </tr>
</table>  

<br><br>
## Train
`./trainer` 폴더에서 진행합니다.

Stage1과 Stage2로 나눠서 학습하는 방식을 사용합니다.
1. Config 수정
	- `./config` 폴더에서 `train_stage1.yaml`, `train_stage2.yaml`, `accelerate_config.yaml`을 하려는 학습에 맞게 수정해줍니다.
		- `train_stage2.yaml`의 `ckpt` 항목에는 반드시 stage1 학습이 종료된 후에 얻은 pth 파일의 경로를 넣어주어야 합니다.
	- `sh ./config/accelerate_config.sh`를 실행하면 원하는 accelerate 환경에 맞는 config를 쉽게 생성할 수 있습니다.

2. Train 실행
	- `./accelerate_train.h`에서 train_config의 path를 수정할 수 있습니다.
	- 옵션 : 
		- `--cfg-path {train_config_path}`
		- `--dryrun(option)`
```
sh accelerate_train.sh --cfg-path {train_config_path}
```

## Evaluate
`./evaluator` 폴더에서 진행
1. Config 수정
	- `salmonn_eval_config.yaml`과 `./config/accelerate_config.yaml` 파일을 수정해 현재 환경에 맞는 evaluate 실행
	- 주로 `config 파일들의 path, batch_size, token` 등을 수정하면 됩니다.
	- `sh ./config/accelerate_config.sh`를 실행해 원하는 환경에 맞느 accelerate config 파일을 쉽게 생성 가능
	- ![[accelerate_config.png|300]]
		- `accelerate_config.yaml` 파일의 내용으로 현재 Single GPU에 맞는 설정으로 되어 있습니다. 만약 Multi GPU를 활용해 분산 학습 및 추론을 하고 싶다면 `#Single GPU` 라인을 주석처리한 뒤 `#Multi GPU`가 적힌 라인을 주석 해제해 사용하면 기본적인 Multi GPU 활용이 가능합니다.

2. Torch-TensorRT로 Compile
```
python tensorrt_aot.py
```
- `tensorrt_aot.py`를 실행하면 `salmonn_eval_config.yaml`에 설정된 모델의 경로와 `batch_size_eval, optimization_level, tensorrt_device`의 config 값들을 가져와 사용합니다.
- evaluate으로 추론 결과를 생성할 때는 현재 하드웨어 스펙에 맞는 batch_size를 지정해 compile을 진행 (ex: `batch_size_eval = 8`)
- evalute_efficiency로 메모리 사용량과 latency를 측정할 때는 `batch_size_eval`을 무조건 `1`로 설정해 `python tensorrt_aot.py`를 실행해야 됩니다.
	- `salmonn_eval_config.yaml`에서 `batch_szie_eval=1` 설정
- 이후 모델은 `./trt_models`폴더에 저장되어 추후 evaluate 과정에서 사용됩니다.

3. Evaluate
```
sh accelerate_eval.sh --mode {submission_mode}
```
- `salmonn_eval_config.yaml`과 `./config/accelerate_config.yaml`에 설정된 config 값들에 따라 다르게 실행될 수 있습니다.
- 현재 `./config/accelerate_config.yaml`은 Single GPU로 설정되어 있으니 위의 Config 과정처럼 Multi GPU 설정으로 바꿀 수 있습니다.
- 2번 과정을 통해 모델을 TensorRT로 Compile해 저장해서 사용하려면 compile할 때 사용한 것과 동일한 `batch_size_eval` 크기를 사용해야 됩니다.
	- ex) `python tensorrt_aot.py`를 실행할 때 `salmonn_eval_config.yaml`의 `batch_size_eval`이 `8`로 설정되어 있다면 이 `batch_size_eval` 값을 그대로 `sh accelerate_eval.sh`를 실행할 때 사용해야 합니다.
- `--mode` 옵션
	- `--mode {submission_asr/submission_aac}`로 사용

4. Evaluate Efficiency
```
python evaluate_efficiency_salmonn.py
```
- 모델의 메모리 사용량과 latency를 측정하는 파일입니다.
- `salmonn_eval_config.sh`의 설정된 config 값들을 사용합니다.
- TensorRT로 compile한 모델을 사용하려면 `python tensorrt_aot.py`를 실행할 때 `salmonn_eval_config.sh`의 `batch_size_eval`을 무조건 `1`로 설정해서 사용해야 합니다.
