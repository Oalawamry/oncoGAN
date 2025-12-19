FROM mambaorg/micromamba:2.4.0-alpine3.21

## Install alpine dependencies
USER root
RUN apk update && \
apk add --no-cache --virtual .build-deps build-base
USER $MAMBA_USER
## Install micromamba environment
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements/environment.yml /tmp/environment.yml
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements/dae_environment.yml /tmp/dae_environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba create -y -n dae -f /tmp/dae_environment.yml && \
    micromamba clean --yes --all && \
    rm /tmp/environment.yml /tmp/dae_environment.yml
## Clean alpine dependencies
USER root
RUN apk del .build-deps
USER $MAMBA_USER
## Copy liftover chain file
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements/hg19ToHg38.over.chain.gz /.liftover/hg19ToHg38.over.chain.gz
## Copy models directory
COPY --chmod=777 --chown=$MAMBA_USER:$MAMBA_USER models/ /oncoGAN/models/
COPY --chmod=777 --chown=$MAMBA_USER:$MAMBA_USER src/* /oncoGAN/
## Update PATH
ENV PATH=$PATH:/oncoGAN
ENV MPLCONFIGDIR=/tmp
## Entrypoint
WORKDIR /home
EXPOSE 5678
# ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "/oncoGAN/launcher.py"]