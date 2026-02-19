export DISPLAY=
export PYOPENGL_PLATFORM=egl
export PANDA_PRC_FILE_DATA="load-display egl\naux-display egl"

python src/megapose/infer.py
