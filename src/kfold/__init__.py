"""KFold biomolecular structure prediction."""

__all__ = ["KFoldRunner"]


def __getattr__(name):
    if name == "KFoldRunner":
        from kfold.runner import KFoldRunner

        return KFoldRunner
    raise AttributeError(name)
