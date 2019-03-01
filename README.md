# Click Models for Mobile Search 

Thanks for visiting.

This project focuses on building click models for mobile search.

It is based on the project by Aleksandr Chuklin (https://github.com/varepsilon/clickmodels).

Published under the BSD license.

# How to install

```default
git clone https://github.com/THUIR/click_model_for_mobile_search.git
cd mobile_click_model
python setup.py install
```
# Models Implemented

- Mobile Click Model (**MCM**): Jiaxin Mao, Cheng Luo, Min Zhang, and Shaoping Ma. 2018. Constructing Click Models for Mobile Search. SIGIR (2018).
- Viewport Time Click Model (**VTCM**): Not yet published.
- Dynamic Bayesian Network (**DBN**) model: Chapelle, O. and Zhang, Y. 2009. A dynamic bayesian network click model for web search ranking. WWW (2009).
- User Browsing Model (**UBM**): Dupret, G. and Piwowarski, B. 2008. A user browsing model to predict search engine click data from past observations. SIGIR (2008).
- Exploration Bias User Browsing Model (**EB-UBM**): Chen, D. et al. 2012. Beyond ten blue links: enabling user click modeling in federated web search. WSDM (2012).
- Dependent Click Model (**DCM**): Guo, F. et al. 2009. Efficient multiple-click models in web search. WSDM (2009).
- A version of User Browsing Model considering different result layouts (**UBM-layout**): Aleksandr Chuklin, Pavel Serdyukov, and Maarten de Rijke. 2013. Using intent information to model user behavior in diversified search. ECIR (2013)



# Format of the Input Data (Click Log)

A small example can be found under ``data/train``. This is a tab-separated file, where each line has 10 elements. A record of a query session is shown as follows:

```
0032202F707D59045B51569FA5952795#1#1533349671#5#0	快递单号查询	0	0.0	["http://www.guoguo-app.com", "https://m.ickd.cn/", "http://m.ickd.cn/", "http://m.kuaidi100.com/", "http://m.kuaidi100.com/", "http://www.chawuliu.cn/", "http://www.17ckd.com/", "http://wap-hint/", "http://www.sogou.com/", "http://m.aikuaidi.cn", "http://wap.yto.net.cn/", "http://www.sf-express.com/mobile/cn/sc/index.html", "http://www.sogou.com/"]	["21194401", "-1", "-1", "-1", "30000909", "-1", "-1", "30010081", "50023801", "-1", "-1", "30000909", "50023901"]	[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]	[30000, 30000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]	[20333, 2055, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]	[345, 157, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
```

1. ``0032202F707D59045B51569FA5952795#1#1533349671#5#0`` contains the hashid and beginning timestamp of the session (currently not used in MCM and VTCM).

2. ``快递单号查询`` is the query of the session.

3. A integer identifier which represent the user's region (currently not used in MCM and VTCM).
4. The probability $P(I = V)$ that the user has a vertical intent $V$ (currently not used in MCM and VTCM).

5. A list of the URLs of the documents that make up SERP (search engine result page).

6. A list with the presentation types (vertical types) of the documents.

7. A list of user clicks.

8. A list of raw viewport time.

9. A list of viewport time weighted by the result height and viewport height.

10. A list of result heights.


# Usage

## Click prediction

You can train and test a click model with given example data:
```default
python test_click_models.py ../data/train ../data/test -m MCM-VPT -N 1 -M 1 -o ../mcm_mr --ignore_no_clicks --ignore_no_viewport --viewport_time
```

- ``-m``: The click model that you would like to run, such as MCM-VPT (VTCM), MCM, and other popular click models.

- ``-N``: The number of files that you would like to use for training in the train dictionary.

- ``-M``: The number of files that you would like to use for test in the test dictionary.

- ``-o``: The path to output dictionary.

- ``--ignore_no_clicks``: Ignore the sessions that don't contain any click.

- ``--ignore_no_viewport``: Ignore the sessions that don't contain viewport time information

- ``--viewport_time``: Read viewport time information of train and test dataset.

- ``-V``: The viewport time model used in VTCM (used with ``-m MCM-VPT``). 

  - ``0``: VTCM$_e$ with log-normal
  - ``1``: VTCM$_e$ with gamma
  - ``2``: VTCM$_e$ with Weibull 
  - ``3``: VTCM$_c$ with log-normal
  - ``4``: VTCM$_c$ with gamma
  - ``5``: VTCM$_c$ with Weibull


## Relevance estimation

You can get the relevance scores estimated by click models based on the outputs of click prediction:
```default
python get_ranking_from_relevance_estimation.py ../mcm_mr/relevance_estimation.txt ../mcm_mr/ranking_relevance_estimation.txt -n 10 -m MCM
```

# Information

## Authors

Yukun Zheng, Tsinghua University 

Jiaxin Mao, Tsinghua University

## Citations


For MCM, please cite:

Jiaxin Mao, Cheng Luo, Min Zhang, Shaoping Ma. Constructing Click Models for Mobile Search. The 41st International ACM SIGIR Conference on Research and Development in Information Retrieval. (SIGIR 2018)


## Contact

Please contact Yukun Zheng by the following email if you have any question:

```
zhengyk13 at gmail.com
```


