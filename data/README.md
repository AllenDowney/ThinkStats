# Data files

## National Survey of Family Growth (NSFG)

These files are NSFG public-use data, or extracts made from them, released by
the National Center for Health Statistics (NCHS):

| File | Contents | Used in |
|---|---|---|
| `2002FemPreg.dct`, `2002FemPreg.dat.gz` | Cycle 6 (2002) pregnancy file | Chapters 1–6, 9, 11, 13, 14 |
| `2002FemResp.dct`, `2002FemResp.dat.gz` | Cycle 6 (2002) female respondent file | Chapters 2, 3, 5, 6 |
| `marriage_nsfg_female.csv.gz` | Marriage variables from female respondent files, ten cycles from 1982 to 2022–2023, made with [MarriageNSFG](https://github.com/AllenDowney/MarriageNSFG); only the 9 columns Chapter 13 uses | Chapter 13 |
| `marriage_nsfg_male.csv.gz` | Male respondents, cycles 2002–2019 (older version, not updated) | not used in the book |
| `1995FemRespData.dat.gz`, `2006_2010_FemResp*` | Respondent files from other cycles, read by functions in `nsfg.py` | not used in the book |

The `.dat.gz` files are gzipped copies of the files NCHS distributes; the
`.dct` files are identical to the NCHS Stata dictionaries. The originals are at
<https://www.cdc.gov/nchs/nsfg/> and
<https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NSFG/>.

### Terms of use

By using these data you agree to the
[NCHS Data User Agreement](https://www.cdc.gov/nchs/policy/data-user-agreement.html).
Users will:

1. Use the data in this dataset for statistical reporting and analysis only.
2. Make no attempt to learn the identity of any person or establishment included in these data.
3. Not link this dataset with individually identifiable data from other NCHS or non-NCHS datasets.
4. Not engage in any efforts to assess disclosure methodologies applied to protect individuals and establishments or any research on methods of re-identification of individuals and establishments.

### Citation

NCHS asks that published work cite the source as "National Center for Health
Statistics, National Survey of Family Growth", with the cycle, and state that
the analyses, interpretations and conclusions are the author's, not NCHS's,
which is responsible only for the initial data release.
