ARG GO_VERSION=1.26
ARG APP=api

FROM golang:${GO_VERSION}-alpine AS build
ARG APP=api
WORKDIR /src
COPY go.mod ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/app ./cmd/${APP}

FROM gcr.io/distroless/static-debian12:nonroot
WORKDIR /app
COPY --from=build /out/app /app/app
EXPOSE 8080 8090
USER nonroot:nonroot
ENTRYPOINT ["/app/app"]
