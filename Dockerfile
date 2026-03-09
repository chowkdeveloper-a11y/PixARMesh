FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel

ARG DOCKER_USER=ubuntu USER_HOME=/workspace

ARG EXTRA_DEPS="libglib2.0-0 libegl1 libgl1 libgomp1 python3-venv passwd"

ARG TOOLS="nvtop tmux htop vim"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install --no-install-recommends -y sudo git ninja-build $TOOLS $EXTRA_DEPS && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    if id -u "$DOCKER_USER" >/dev/null 2>&1; then \
        echo "User $DOCKER_USER exists; setting home to $USER_HOME"; \
        mkdir -p "$USER_HOME"; \
        usermod -d "$USER_HOME" -m "$DOCKER_USER"; \
    else \
        adduser --disabled-password --gecos '' --home "$USER_HOME" "$DOCKER_USER"; \
    fi; \
    adduser "$DOCKER_USER" sudo; \
    chown -R "$DOCKER_USER:$DOCKER_USER" "$USER_HOME"

ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0 12.0+PTX" FORCE_CUDA=1

ADD requirements.txt requirements-no-iso.txt /tmp/

RUN pip config set global.break-system-packages true && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    MAX_JOBS=24 pip install --no-cache-dir --no-build-isolation -r /tmp/requirements-no-iso.txt

RUN git clone https://github.com/NVlabs/EdgeRunner.git /opt/EdgeRunner && \
    cd /opt/EdgeRunner/meto && \
    pip install --no-build-isolation --config-settings editable_mode=compat -e .

USER $DOCKER_USER
