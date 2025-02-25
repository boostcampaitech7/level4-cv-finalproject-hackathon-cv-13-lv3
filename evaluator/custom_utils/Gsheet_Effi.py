import os

import gspread
from gspread.exceptions import WorksheetNotFound
from gspread_formatting import *
from dotenv import dotenv_values

def Gsheet_param(cfg, *args):
    # env 파일 불러오기
    env_path = "/data/env/.env"
    env = dotenv_values(env_path)

    # 서비스 연결
    gc = gspread.service_account(env['JSON_PATH'])

    # url에 따른 spread sheet 열기
    doc = gc.open_by_url(env['URL_Efficiency'])

    # 저장할 변수 dict 선언
    param_dict = dict()

    # User 명
    param_dict['user'] = os.path.abspath(__file__).split("/")[2]

    # for idx, (key, value) in enumerate(cfg.items()):
    #     if idx < 4:

    #         pass
    #     else :
    param_dict['backbone'] = cfg.config.model.llama_path
    param_dict['Encoder_Speech'] = cfg.config.model.whisper_path
    param_dict['Encoder_Audio'] = cfg.config.model.beats_path
    param_dict['average_memory_usage(GB)'] = args[0]/1024**3
    param_dict['average_inference_time(sec)'] = args[1]
    param_dict['average_ttft(sec)'] = args[2]
    param_dict['average_tpot(sec)'] = args[3]


    # sheet에 추가하기 위해서 값들을 list로 저장
    params = [param_dict[k] for k in param_dict]

    # sheet가 없는 경우 Head Row를 구성하기 위해서 Col 명을 list로 저장
    cols = [k.capitalize() for k in param_dict]
    
    try:
        # 워크시트가 있는지 확인
        worksheet = doc.worksheet(cfg.config.project_name)
    except WorksheetNotFound:
        # 워크시트가 없으면 새로 생성
        worksheet = doc.add_worksheet(title=cfg.config.project_name, rows="1000", cols="30")
        # Col 명 추가
        worksheet.append_rows([cols])

        # Header Cell 서식 
        header_formatter = CellFormat(
            backgroundColor=Color(0.9, 0.9, 0.9),
            textFormat=TextFormat(bold=True, fontSize=12),
            horizontalAlignment='CENTER',
        )
        
        # Header의 서식을 적용할 범위
        header_range = f"A1:{chr(ord('A') + len(cols) - 1)}1"

        # Header 서식 적용
        format_cell_range(worksheet, header_range, header_formatter)

        # Header Cell의 넓이 조정
        for idx, header in enumerate(cols):
            column_letter = chr(ord('A') + idx)
            width = max(len(header)*10+20,80)
            set_column_width(worksheet, column_letter, width)

        print(f"'{cfg.config.project_name}' 워크시트가 생성되었습니다.")

    # 실험 인자를 작성한 worksheet
    worksheet = doc.worksheet(cfg.config.project_name)

    # 실험 인자 worksheet에 추가
    worksheet.append_rows([params])

    # 현재 작성하는 실험 인자들 Cell의 서식
    # 노란색으로 하이라이트
    row_formatter = CellFormat(
        backgroundColor=Color(1, 1, 0),
        textFormat=TextFormat(fontSize=12),
        horizontalAlignment="CENTER"
    )

    # 이전 작성 실험인자들 배경색 원상복구
    rollback_formatter = CellFormat(
        backgroundColor=Color(1.0, 1.0, 1.0)
    )
    
    # 마지막 줄에만 하이라이팅이 들어가야 하므로 마지막 row 저장
    last_row = len(worksheet.get_all_values())
    row_range = f"A{last_row}:{chr(ord('A') + len(cols) - 1)}{last_row}"
    rollback_range = f"A{last_row - 1}:{chr(ord('A') + len(cols) - 1)}{last_row - 1}"
    
    # 헤더셀의 서식이 초기화되는 것을 방지하기 위한 조건문
    if last_row - 1 != 1:
        format_cell_range(worksheet, rollback_range, rollback_formatter)
    
    format_cell_range(worksheet, row_range, row_formatter)