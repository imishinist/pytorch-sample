
.PHONY: save-package
save-package:
	pip freeze > requirements.txt

.PHONY: install-package
install-package:
	pip install -r requirements.txt
