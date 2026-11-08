# OSS-specific variables and commands
DOCKER_TAG ?= dev
DOCKER_IMAGE_OSS=oss_pytests:${DOCKER_TAG}
INFRA_PROVIDER ?= AWS

CACHE_REF ?=
CACHE_BRANCH ?= main
CACHE_NAME ?= oss-sdk
comma := ,
_CACHE_FROM = $(if $(CACHE_REF),--cache-from type=registry$(comma)ref=$(CACHE_REF):cache-$(CACHE_NAME)-$(CACHE_BRANCH) --cache-from type=registry$(comma)ref=$(CACHE_REF):cache-$(CACHE_NAME)-main)
_CACHE_TO = $(if $(CACHE_REF),--cache-to type=registry$(comma)ref=$(CACHE_REF):cache-$(CACHE_NAME)-$(CACHE_BRANCH)$(comma)mode=max$(comma)compression=zstd$(comma)compression-level=3$(comma)force-compression=true)

build-oss-ci:
	docker buildx build \
		--output=type=docker \
		--provenance=false \
		$(_CACHE_FROM) \
		$(_CACHE_TO) \
		-f oss_pytests/Dockerfile \
		-t $(DOCKER_IMAGE_OSS) \
		oss_pytests

start-docker-oss:
	docker run \
		--rm \
		-e INFRA_PROVIDER=${INFRA_PROVIDER} \
		-e URL=${URL} \
		-e ESM_API_KEY \
		-e CUDA_VISIBLE_DEVICES="" \
		--name=$(USER)-oss_pytests \
		--network=host \
		${DOCKER_IMAGE_OSS}
