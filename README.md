docker build -t deep-philosopher . &&
docker run --rm -it -p 8080:8080 deep-philosopher


gcloud app deploy --verbosity=debug