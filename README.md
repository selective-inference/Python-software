# The selection project

This project contains software for selective inference, with
emphasis on selective inference in regression. 

Some key references:

* `A significance test for the lasso`: http://arxiv.org/abs/1301.7161
* `Tests in adaptive regression via the Kac-Rice formula`: http://arxiv.org/abs/1308.3020
* `Post-selection adaptive inference for Least Angle Regression and the Lasso`:  http://arxiv.org/abs/1401.3889
* `Exact post-selection inference with the lasso`:  http://arxiv.org/abs/1311.6238
* `Exact Post Model Selection Inference for Marginal Screening`: http://arxiv.org/abs/1402.5596

Install
-------


```
git submodule init # travis_tools and C-software
git submodule update
pip install -r requirements.txt
python setup.py install
```
