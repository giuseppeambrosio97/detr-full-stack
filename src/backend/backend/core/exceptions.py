class ModelNotExportedError(Exception):
    def __init__(self, model_name: str) -> None:
        super().__init__(f"Export is required for the model '{model_name}'.")
