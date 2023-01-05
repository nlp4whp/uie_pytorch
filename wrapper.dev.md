# UIE Wrapper

## Schema

``` python
schema = {
    'NER': List[str]  # [label]
    'Rel': Dict[str, List[str]]  # Dict[head_label, List[rel_label]]
}
```

## Develop

``` sh
poetry init --name uie-wrapper
poetry env ues python
poetry shell
poetry add -D pytest rich comm-utils six colorlog colorama tqdm
poetry add -D jupyterlab jupyter-resource-usage jupyterlab-code-formatter ipywidgets isort black
poetry add -D numpy torch transformers sentencepiece faster-tokenizer
poetry add -D onnx onnxruntime-gpu onnxconverter_common

# build as xxx.whl.
poetry build -f wheel
```
## Usage Command

#### PyTest

``` sh
poetry run pytest tests/test_xxx.py --disable-pytest-warnings -s
```

#### Annotation with [doccano](https://github.com/doccano/doccano)

``` sh
docker pull doccano/doccano
mkdir -p data
#    --user $(id -u):$(id -g) \
docker run --rm -it --name doccano-uie \
    --cpus=4 \
	--memory=4g \
    -p 9082:8000 \
    -e "ADMIN_USERNAME=gpu3" \
    -e "ADMIN_EMAIL=admin@example.com" \
    -e "ADMIN_PASSWORD=gpu3" \
    -v data:/data \
    doccano/doccano
```

#### Jupyter Notebook ENV

``` sh
jupyter lab --ip='0.0.0.0' \
    --port='9081' \
    --allow-root --NotebookApp.token='' --NotebookApp.password=''
```
