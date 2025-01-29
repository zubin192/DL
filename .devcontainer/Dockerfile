FROM ubuntu:22.04

# Install system packages
RUN apt-get -q update \
  && apt-get -q -y upgrade \
  && apt-get -q install --no-install-recommends -y \
  git \
  lsb-release \
  curl \
  build-essential \
  python3 \
  python3-pip \
  python3-opengl \
  sudo \
  # libboost-all-dev \
  && apt-get autoremove -y \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*

# Install Python packages
# RUN python3 -m pip install mypy
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Configure a new non-root user
ARG USERNAME=project
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
  && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  && usermod -a -G dialout $USERNAME \
  && echo "source /usr/share/bash-completion/completions/git" >> /home/$USERNAME/.bashrc


WORKDIR /workspace

CMD [ "bash" ]
