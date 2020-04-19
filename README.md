# corona_models
Simplistic models of COVID-19, has a SEIR and SEIRC models.

The toys allow to model social distancing measures, to get an idea of how
it this could affect the outcomes of the epidemic.

---

To try needs python 3.X (3.7 recommended)

Libraries needed
----------------

pandas, numpy, scipy, matplotlib

---

The rest is just:

   python -i models.py

It will by default plot some example results for a situation where the
government puts a quarantine 30 days after the first detected case. The
quarantine lasts 30 days. After this, people can either learn and change
a lot their social behaviour, so half what they used to interact, or they could
change but not so drastically. All under the SEIR model.

---

TODO/FIXME
----------
    - [ ] Better estimation of Betas.
    - [ ] Add estimated deaths from recovered/removed.
    - [ ] Add available beds in ICUs as column, to be plotted always.
    - [ ] Prettyplots?
