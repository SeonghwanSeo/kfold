/usr/bin/git -C /tmp/icl_mseok/kfold-main-schedule diff -- \
  /tmp/icl_mseok/kfold-main-schedule/configs/model/module/structure_module/ecsi.yaml \
  /tmp/icl_mseok/kfold-main-schedule/src/kfold/model/modules/structure_module/kfold_ecsi.py

/home/icl_mseok/.local/bin/ruff format \
  /tmp/icl_mseok/kfold-main-schedule/src/kfold/model/modules/structure_module/kfold_ecsi.py

/home/icl_mseok/.local/bin/ruff check \
  /tmp/icl_mseok/kfold-main-schedule/src/kfold/model/modules/structure_module/kfold_ecsi.py
