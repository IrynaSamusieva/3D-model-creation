# RTMW-m → SMPLify-X → SMPL-X

Пайплайн определяет 133 ключевые точки человека на фотографии, подгоняет к ним модель SMPL-X и экспортирует 3D-сетку в OBJ.

## Запуск

Из PowerShell в корне проекта:

```powershell
Set-Location D:\Bakalarka
& .\venv\Scripts\python.exe .\scripts\run_rtmw_smplifyx.py `
  --smplifyx-repo 'D:\Bakalarka\smplify-x-master\smplify-x-master' `
  --image 'D:\Bakalarka\image.avif' `
  --output 'D:\Bakalarka\results\image'
```

Для другого снимка замените значение `--image`. Если рабочие Python-интерпретаторы находятся по другим путям, передайте их через `--mmpose-python` и `--smplifyx-python`.

## Результаты

- Ключевые точки RTMW: `results/<имя>/rtmw/`.
- SMPLify-X: `results/<имя>/smplifyx/`.
- Финальная сетка и параметры: `results/<имя>/avatar/`.

## Зависимости и модели

Для RTMW требуются PyTorch с подходящей CUDA-сборкой, MMPose, MMCV и MMEngine. Для чтения AVIF нужен `pillow-avif-plugin`. Этап SMPLify-X также требует пакеты `smplx` и `human_body_prior`, файлы модели SMPL-X и веса VPoser.

Клонированный SMPLify-X, веса RTMW, модели SMPL-X/VPoser и результаты обработки не входят в этот репозиторий. Подготовьте их отдельно и укажите пути в параметрах запуска. Перед использованием проверьте лицензии моделей и SMPLify-X.
