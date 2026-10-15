# Collaborative testing files

사용자 지시에 따라 실험 소스는 기존 `collaborative_test/`에 유지합니다. 생성된 결과·로그·캐시의 실제 저장 위치는 `generated/collaborative_test/`입니다. 기존 디렉터리 경로에는 junction, 개별 결과 파일 경로에는 NTFS hard link를 유지하여 실행·import·Git 기록을 보존합니다. 개별 파일은 복사본 두 개가 아니라 같은 파일 ID를 공유하는 호환 경로이며, 이동 전후 SHA256 및 파일 ID를 검증했습니다. 생성 파일의 중앙 경로는 로컬 정리용이며 Git 공유는 기존 연구 기록 경로를 유지합니다.

새 실험의 결과 경로는 파일을 쓰기 전에 다음 명령으로 준비합니다. 이 명령은 실험 소스를 이동하지 않습니다.

```powershell
.\collaborative_testing\prepare_output.ps1 -Experiment SW_0166_fixed_support_ari_diagnosis
```

추가 정리: 기존 실험 코드 아래의 캐시·임시 fixture 120개 폴더(703개 파일)를 `temporary/canonical_test_generated/`로 이동했습니다. 이동 전후 모든 파일의 SHA256 일치를 확인했습니다. 원래 위치와 새 위치는 `generated_organization_manifest_20261014.json`에 기록했습니다.

실험 도중 생성된 로컬 파일을 이 폴더에 모았습니다.

| 위치 | 내용 |
|---|---|
| `collaborative_private_runtime_20261008/` | 현재 사용하는 배포 묶음·검증 스크립트·GitHub 공유 검토 자료 |
| `artifacts/` 또는 `legacy_workspace/artifacts/` | 기존 시각화·분석 산출물 |
| `results/`, `server_results/` | 저장된 결과 그림·mask |
| `server_train/` | 서버 학습 보조 스크립트 |
| `legacy_workspace/`, `legacy_repo/`, `legacy_desktop/` | 이전에 여러 위치로 흩어졌던 파일. 원래 출처별로 보존 |
| `delivery_bundles/` | 이전 서버 전송용 압축 파일 |
| `temporary/`, `cache/` | 테스트 임시 파일과 캐시 |

이동 전후 파일 목록과 SHA256은 `organization_manifest_20261014.json` 및 검증 기록에 남겼습니다. 원래 경로가 들어 있는 과거 배포 기록은 당시 기록으로 보존하며, 이동표에서 새 위치를 찾을 수 있습니다.

앞으로 로컬 검증은 저장소 루트에서 다음처럼 실행합니다. 임시 폴더는 실행별로 분리하고 테스트 종료 후 보존합니다.

```powershell
.\collaborative_testing\run_tests.ps1 -Python 'C:/Users/SangWoo Kim/anaconda3/python.exe' -Framework unittest collaborative_test.SW_0164_foreground_preserving_partition.test_foreground_signed
```

서버 학습 보조 스크립트의 새 모듈 경로는 `collaborative_testing.server_train`입니다. 예: `python -m collaborative_testing.server_train.train_input_layer --help`.
