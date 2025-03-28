# 스도쿠 게임 구현 프로젝트

이 프로젝트는 다양한 크기의 스도쿠 퍼즐을 해결하기 위한 알고리즘을 구현한 것입니다.

## 구현된 스도쿠 종류

1. 2x3 스도쿠 (`2x3_sudoku.py`)
   - 6x6 크기의 스도쿠 퍼즐
   - 2x3 크기의 블록으로 구성

2. 4x4x4 스도쿠 (`4x4x4_sudoku.py`)
   - 16x16 크기의 스도쿠 퍼즐
   - 4x4 크기의 블록으로 구성

3. EOR 스도쿠 (`eor_sudoku.py`)
   - XOR 연산을 활용한 스도쿠 변형
   - 인접한 셀 간의 XOR 관계를 만족해야 함

## 사용 방법

각 스도쿠 파일은 독립적으로 실행할 수 있습니다. 예를 들어:

```bash
python 2x3_sudoku.py
python 4x4x4_sudoku.py
python eor_sudoku.py
```

## 프로젝트 구조

```
sudoku/
├── 2x3_sudoku.py      # 2x3 스도쿠 구현
├── 4x4x4_sudoku.py    # 4x4x4 스도쿠 구현
├── eor_sudoku.py      # EOR 스도쿠 구현
├── report.tex         # 프로젝트 보고서
└── data/             # 데이터 디렉토리
    └── 스크린샷 2025-02-25 08.27.24.png
```

## 요구사항

- Python 3.x
- NumPy (선택사항)

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 