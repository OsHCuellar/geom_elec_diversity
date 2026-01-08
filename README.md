# geom-elec-diversity

Quantifying and controlling geometric and electronic information content in datasets.

## Representations (SPAHM(a,b), SLATM) location

**All representations are available in the form of `.npy` array on `lcmdlc3` @**
```bash
/home/calvino/yannick/ENN-SPAHM/qm7/Xarray
```
Filenames follow the same formating:
`X_$REP_$STATE.npy`

where:
- `$REP` is one of [spahm-a, spahm-b, spahm-a-global, spahm-b-global, slatm]
- `$STATE` is one of [neutral, vertical, relaxed]
- `$STATE` == neutral_0 correspond to neutral,singlet compounds treated as open-shell system (`spin=0`) for size consistency !


## Some reference papers

- https://doi.org/10.1002/hbm.23471
- https://doi.org/10.1186/s13321-019-0391-2
- https://doi.org/10.1002/hbm.23471
- https://doi.org/10.1021/ar010048x
- https://doi.org/10.1186/s13321-016-0176-9
- https://doi.org/10.1186/s13321-024-00883-4
- https://doi.org/10.1021/acs.jcim.5c00175
