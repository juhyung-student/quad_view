# quad_view

사진 또는 동영상 하나를 네 방향으로 회전해 검은 화면에 표시하는 Python 데스크톱 프로그램입니다. 동영상은 OpenCV에서 한 번만 읽고 같은 프레임을 네 방향으로 복제하므로 네 화면이 프레임 단위로 동기화됩니다. 영상 그룹은 화면 맞춤 크기에서 80%로 표시됩니다.

## 요구 환경

- Windows
- Python 3.12

## 설치 및 실행

프로젝트 폴더에서 다음 명령을 실행합니다.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python quad_view.py
```

PowerShell 실행 정책 때문에 가상 환경 활성화가 제한되면 다음처럼 인터프리터를 직접 실행할 수 있습니다.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe quad_view.py
```

## 지원 파일

- 사진: PNG, JPG, JPEG, WEBP
- 동영상: MP4, AVI, MOV, MKV

동영상의 프레임 속도를 읽어 재생하며, FPS 정보를 사용할 수 없으면 30 FPS를 적용합니다. 동영상은 끝에 도달하면 자동으로 처음부터 반복 재생됩니다.

## 사진 2.5D 입체 효과

기본 GUI 의존성과 깊이 추정 의존성은 분리되어 있습니다. 사진의 `입체 효과 켜기`를 사용하려면 모델용 패키지를 추가 설치합니다. NVIDIA GPU/CUDA가 없는 환경에서는 CPU 전용 PyTorch를 설치합니다.

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-depth.txt
```

CPU는 큰 사진에서 분석 시간이 길 수 있습니다. 지정 모델 [`depth-anything/Depth-Anything-V2-Small-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)의 Hugging Face 저장소 사용량은 약 99.2MB이며, 첫 분석 때 인터넷 연결과 모델 캐시 저장 공간이 필요합니다. 공식 모델 카드는 Apache-2.0 라이선스를 표시합니다. 깊이값은 실제 거리 단위가 아닌 단안 상대 깊이입니다.

사진을 불러온 뒤 `입체 효과 켜기`를 선택하면 깊이를 분석하고 네 시점 이미지를 만듭니다. `시점 간격`과 `초점면`을 바꾸면 저장된 깊이 지도를 재사용하므로 모델을 다시 실행하지 않습니다. `깊이 반전`은 가까움/멀음 해석을 뒤집습니다. `깊이 지도 미리보기`에서 원본, 흑백 깊이, 색상 깊이를 확인할 수 있습니다. `합성 입체 장면`은 원·사각형·삼각형의 지정된 깊이와 시차 방향을 모델 다운로드 없이 확인하는 테스트입니다. `원본 모드로 돌아가기`를 누르면 기존 회전 이미지 모드로 복귀합니다.

시점 이동은 최대 32px로 제한됩니다. 한 장에 보이지 않는 가려진 뒷면은 복원할 수 없으며, 재투영 뒤 생긴 작은 홀은 주변 픽셀을 제한적으로 인페인팅합니다. 따라서 실제 3D 복원은 아니며 사진별 깊이 오류나 가림 경계 아티팩트가 생길 수 있습니다. 동영상의 AI 입체 변환은 이번 단계에서 적용하지 않습니다.

## 투영 보정

미디어 화면의 `투영 보정`을 켜고 강도 슬라이더를 조정합니다. 기본 강도는 0이며, `보정 방향 반전`으로 반대 방향의 곡률 프로필을 비교할 수 있습니다. 보정은 최종 회전된 네 이미지에 같은 원칙으로 적용됩니다.

`보정 테스트 패턴`을 누르면 격자, 가로·세로 기준선, 모서리 표시와 하단 확인선이 나타납니다. `미디어 보기`로 선택한 사진이나 영상 화면으로 돌아옵니다. 실제 홀로그램 피라미드의 반사 곡률은 장치마다 다르므로, 투영하면서 보정 강도와 방향을 조정해야 합니다. 화면 미리보기만으로 물리적 투영 결과를 보증하지 않습니다.