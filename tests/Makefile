# OSS-specific variables and commands
DOCKER_TAG ?= dev
DOCKER_IMAGE_OSS=oss_pytests:${DOCKER_TAG}
INFRA_PROVIDER ?= AWS

build-oss-ci:
	docker build \
		--output=type=docker \
		-f oss_pytests/Dockerfile \
		-t $(DOCKER_IMAGE_OSS) \
		oss_pytests

start-docker-oss:
	docker run \
		--rm \
		-e INFRA_PROVIDER=${INFRA_PROVIDER} \
		-e URL=${URL} \
		-e ESM_API_KEY=${ESM_API_KEY} \
		--name=$(USER)-oss_pytests \
		--network=host \
		${DOCKER_IMAGE_OSS}
