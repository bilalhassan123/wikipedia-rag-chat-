# Multi-stage frontend image:
#   1. build — Node-based bundle production
#   2. serve — nginx alpine, ~25 MB, with /api reverse-proxy to the backend

FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY . .
RUN npm run build

FROM nginx:alpine AS serve
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
    CMD wget -q --spider http://localhost/ || exit 1
