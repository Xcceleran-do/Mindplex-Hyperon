FROM node:22-alpine AS build

WORKDIR /app

ARG VITE_API_BASE_URL=
ARG VITE_BYPASS_INGESTION=false
ARG VITE_MAX_VIS_ARTICLES=1500
ARG VITE_DEFAULT_CHAIN_DEPTH=3

ENV VITE_API_BASE_URL=${VITE_API_BASE_URL} \
    VITE_BYPASS_INGESTION=${VITE_BYPASS_INGESTION} \
    VITE_MAX_VIS_ARTICLES=${VITE_MAX_VIS_ARTICLES} \
    VITE_DEFAULT_CHAIN_DEPTH=${VITE_DEFAULT_CHAIN_DEPTH}

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run typecheck && npm run build

FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
