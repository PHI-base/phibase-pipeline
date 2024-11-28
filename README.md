# PHI-base pipeline

Python package and command-line application for cleaning and releasing
version 5 of the Pathogen-Host Interactions Database (PHI-base).
Supported release formats include:

-   a JSON file format that combines data from version 4 and version 5
    of PHI-base, and

-   several tabular export formats that are intended for loading by the
    Ensembl databases.

## Installation

Install the latest release from GitHub:

    python -m pip install 'phibase_pipeline@git+https://github.com/PHI-base/phibase-pipeline.git@0.1.0'

Or install the latest commit on the `main` branch:

    python -m pip install 'phibase_pipeline@git+https://github.com/PHI-base/phibase-pipeline.git@main'

## Usage

### JSON release format

To generate a cleaned and validated version of the spreadsheet that
contains the PHI-base 4 dataset, use the following command:

    python -m phibase_pipeline zenodo PHIBASE_CSV CANTO_JSON OUTFILE

Explanation of arguments:

-   `PHIBASE_CSV`: the path to an export of data from PHI-base version
    4, stored in a CSV file. These files can be downloaded from the
    [PHI-base/data](https://github.com/PHI-base/data/tree/master/releases)
    repository.

-   `CANTO_JSON`: the path to an export of approved curation sessions
    from the PHI-Canto curation tool, stored in a JSON file.

-   `OUTFILE`: the destination path for the combined JSON file produced
    by the pipeline.

### Ensembl release format

To generate CSV files that can be loaded into the Ensembl interactions
database, run the following command:

    python -m phibase_pipeline ensembl PHIBASE_CSV CANTO_JSON UNIPROT_DATA DIR

The command will produce three CSV files in the directory specified by
`DIR`:

-   **phibase4_interactions_export.csv**: an export of interactions from
    PHI-base version 4.

-   **phibase5_interactions_export.csv**: an export of interactions from
    curation sessions in the PHI-Canto curation tool.

-   **phibase_amr_export.csv**: an export of interactions between
    pathogen genes and antimicrobial chemicals. Currently these
    interactions are sourced from curation done with the PHI-Canto
    curation tool.

Explanation of arguments:

-   `PHIBASE_CSV`: the path to an export of data from PHI-base version
    4, stored in a CSV file. These files can be downloaded from the
    PHI-base/data repository.

-   `CANTO_JSON`: the path to an export of approved curation sessions
    from the PHI-Canto curation tool, stored in a JSON file.

-   `UNIPROT_DATA`: the path to a file containing data about genes and
    proteins, retrieved from the UniProt Knowledgebase (UniProtKB). This
    file is created by downloading the results of a query to the
    [UniProtKB ID mapping](https://www.uniprot.org/id-mapping) service
    as TSV format.

-   `DIR`: the destination directory for the CSV files created by the
    pipeline.

## UniProt data file format

The file passed to the `UNIPROT_DATA` command-line argument expects the
following column names, in the following order:

-   **From**: the UniProtKB accession number used in the ID mapping
    query.
-   **Entry**: the UniProtKB accession number for the protein.
-   **Organism**: the scientific name of the organism to which the
    protein belongs.
-   **Organism (ID)**: the [NCBI
    Taxonomy](https://www.ncbi.nlm.nih.gov/taxonomy) ID of the organism.
-   **Taxonomic lineage (Ids)**: the taxonomic lineage of the organism,
    containing ID numbers and ranks.
-   **Ensembl**: a gene ID from the [Ensembl
    Genomes](https://ensemblgenomes.org/) database.
-   **EnsemblBacteria**: a gene ID from the [Ensembl
    Bacteria](https://bacteria.ensembl.org/index.html) database.
-   **EnsemblFungi**: a gene ID from the [Ensembl
    Fungi](https://fungi.ensembl.org/index.html) database.
-   **EnsemblMetazoa**: a gene ID from the [Ensembl
    Metazoa](https://metazoa.ensembl.org/index.html) database.
-   **EnsemblPlants**: a gene ID from the [Ensembl
    Plants](https://plants.ensembl.org/index.html) database.
-   **EnsemblProtists**: a gene ID from the [Ensembl
    Protists](https://protists.ensembl.org/index.html) database.

To generate a valid file, use the [UniProtKB ID
mapping](https://www.uniprot.org/id-mapping) service to query one or
more UniProtKB accession numbers. The ‘From database’ should be
‘UniProtKB AC/ID’ and the ‘To database’ should be ‘UniProtKB’ (this is
the default setting).

Then use the Download link on the results page, and use the ‘Customize
columns’ field to set the following columns in the following order:

-   Organism
-   Organism (ID)
-   Taxonomic lineage (Ids)
-   Ensembl
-   EnsemblBacteria
-   EnsemblFungi
-   EnsemblMetazoa
-   EnsemblPlants
-   EnsemblProtists

Alternatively, use the following URL, replacing the `{id}` placeholder
with the ID of your ID mapping job.

`https://rest.uniprot.org/idmapping/uniprotkb/results/stream/{id}?fields=accession%2Corganism_name%2Corganism_id%2Clineage_ids%2Cxref_ensembl%2Cxref_ensemblbacteria%2Cxref_ensemblfungi%2Cxref_ensemblmetazoa%2Cxref_ensemblplants%2Cxref_ensemblprotists&format=tsv`

Note that the `stream` endpoint returns chunks of 500 at a time, and
requires pagination. See [here](https://www.uniprot.org/help/pagination)
for more instructions.

## License

The `phibase_pipeline` package is distributed under the terms of the
[MIT](https://spdx.org/licenses/MIT.html) license.
