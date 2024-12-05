## Add package

```shell
poetry add <package_name>
```

## Remove  
```shell
poetry remove <package_name>
```

## Install dependencies
```shell
poetry install
```

## Update dependencies
```shell
poetry update
```

## Run
```shell
poetry run <command>
```

## Streamlit run
```shell
poetry run streamlit run compfigurator\steamlit_app.py
```

## Create virtual environment
```shell
poetry env use python3.8
```

## Activate virtual environment
```shell
poetry shell
```

## Deactivate virtual environment
```shell
exit
```

## Run raw Expert System
```shell
poetry run python compfigurator\expert_system.py
```

---
> NOTE:
> You would encounter an error on a package from experta. To fix this follow __one__ of the below steps:
> 1. Change the package import in your venv
> 2. Use python version 3.8 (for which this poetry python version should be changed in pyproject.toml)
