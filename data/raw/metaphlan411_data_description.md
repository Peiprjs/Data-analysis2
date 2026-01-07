## Notes: MetaPhlAn 4 TSV Format Description

MetaPhlAn 4.1.1 produces a **tab-separated values (TSV)** file reporting the taxonomic profiles of the microbial communities analysed.

#### File Structure and Contents

-   The file begins with a comment line starting with `#` that describes the database version used to generate the profile.
-   The next line contains the column headers:
    -   `clade_name`: the taxonomic path for each clade.
    -   One column per sample.
-   Each cell under a sample column contains a **relative abundance value**, typically expressed as a percentage. For example, `24.3` means that clade represents approximately 24.3% of the microbial community in that sample.

#### How the Table Is Organized

-   Each row corresponds to one taxonomic feature (a clade).
-   Rows are ordered from broad taxonomic levels down to specific units.
-   Higher taxonomic levels give cumulative abundances of all more specific clades nested within them — for example, a genus-level clade includes the sum of all SGBs within that genus.

#### Taxonomic Paths in `clade_name`

Each `clade_name` entry encodes a full taxonomic lineage using a standard prefix system:

```         
k__Kingdom|p__Phylum|c__Class|o__Order|f__Family|g__Genus|s__Species|t__SGBxxxxx
```

-   `k__` = Kingdom\
-   `p__` = Phylum\
-   `c__` = Class\
-   `o__` = Order\
-   `f__` = Family\
-   `g__` = Genus\
-   `s__` = Species\
-   `t__` = Terminal (in MetaPhlAn 4 this is a **Species-Level Genome Bin**, or **SGB**)

#### Hint: Selecting Features at a Given Level

You may want to select features at one (or more) specific taxonomic levels for your analysis.

For example, to extract **genus-level** features: - Keep clade names that **contain `|g__`** but **do not contain `|s__` or `|t__`**
