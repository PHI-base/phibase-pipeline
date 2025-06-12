# SPDX-FileCopyrightText: 2023-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT

import functools
import pprint
import re
from collections import defaultdict

import jsonschema

from phibase_pipeline import loaders


def _validate(predicate, error_message):
    if not predicate:
        print(error_message)


def validate_allele_ids(alleles):
    for allele_id, allele in alleles.items():
        primary_id = allele['primary_identifier']
        _validate(
            primary_id == allele_id,
            f"allele primary ID does not match allele ID:\n{primary_id}\n{allele_id}",
        )


def validate_allele_primary_ids(alleles):
    for allele_id, allele in alleles.items():
        _validate(
            allele_id == allele['primary_identifier'],
            f"allele ID {allele_id} does not match primary identifier",
        )


def validate_classified_organisms(organisms):
    for taxon_id, organism in organisms.items():
        role = organism.get('role')
        _validate(
            role is not None,
            f"organism {organism['full_name']} [{taxon_id}] has no role",
        )
        _validate(
            role != 'unknown',
            f"organism {organism['full_name']} [{taxon_id}] has unknown role",
        )


def validate_disease_ids(annotations):
    pattern = re.compile(r'^PHIDO:\d{7}$')
    for annotation in annotations:
        if annotation['type'] == 'disease_name':
            disease_id = annotation['term']
            _validate(
                pattern.match(disease_id),
                f"invalid disease ID: {disease_id}",
            )


def validate_feature_session_ids(feature_type, pattern, features, session_id):
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    for feature_id in features:
        match = pattern.search(feature_id)
        if not match:
            raise ValueError(f'session ID not found in {feature_type} ID: {feature_id}')
        feature_session_id = match.group(1)
        _validate(
            feature_session_id == session_id,
            f"{feature_type} ID {feature_id} does not match session ID {session_id}",
        )


def validate_gene_ids(genes):
    for gene_id, gene in genes.items():
        organism = gene['organism']
        uniprot_id = gene['uniquename']
        expected_gene_id = ' '.join((organism, uniprot_id))
        _validate(
            gene_id == expected_gene_id,
            f"gene ID does not match gene properties:\n{gene_id}\n{expected_gene_id}",
        )


def validate_gene_references(genes, organisms):
    organism_names = set(organism['full_name'] for organism in organisms.values())
    for gene_id, gene in genes.items():
        organism = gene['organism']
        _validate(
            organism in organism_names,
            f"invalid organism reference in gene {gene_id} ({organism})",
        )


def validate_one_species_per_genotype(genotypes, session):
    alleles = session['alleles']
    genes = session['genes']
    taxon_of_organism = {
        organism['full_name']: int(taxon_id)
        for taxon_id, organism in session['organisms'].items()
    }
    for genotype_id, genotype in genotypes.items():
        genotype_taxon_id = genotype['organism_taxonid']
        for locus in genotype['loci']:
            for locus_allele in locus:
                allele_id = locus_allele['id']
                allele = alleles[allele_id]
                gene_id = allele['gene']
                gene = genes[gene_id]
                organism_name = gene['organism']
                taxon_id = taxon_of_organism[organism_name]
                _validate(
                    taxon_id == genotype_taxon_id,
                    (
                        f"taxon ID {taxon_id} of gene {gene['uniquename']}"
                        f" does not match taxon ID {genotype_taxon_id} of genotype {genotype_id}"
                    ),
                )


def validate_referenced_alleles(genotypes, alleles):
    allele_ids = alleles.keys()
    for genotype_id, genotype in genotypes.items():
        for locus in genotype['loci']:
            for locus_allele in locus:
                allele_id = locus_allele['id']
                _validate(
                    allele_id in allele_ids,
                    f"{genotype_id} locus allele ID not in session alleles: {allele_id}",
                )


def validate_referenced_genes(alleles, genes):
    gene_ids = genes.keys()
    for allele_id, allele in alleles.items():
        gene_id = allele['gene']
        _validate(
            gene_id in gene_ids, f"{allele_id} gene ID not in session genes: {gene_id}"
        )


def validate_referenced_genotypes(metagenotypes, genotypes):
    genotype_ids = genotypes.keys()
    for metagenotype_id, metagenotype in metagenotypes.items():
        for k in ('pathogen_genotype', 'host_genotype'):
            genotype_id = metagenotype[k]
            _validate(
                genotype_id in genotype_ids,
                f"{metagenotype_id} genotype ID not in session genotypes: {genotype_id}",
            )


def validate_referenced_organisms(genotypes, organisms):
    organism_ids = organisms.keys()
    for genotype_id, genotype in genotypes.items():
        organism_id = str(genotype['organism_taxonid'])
        _validate(
            organism_id in organism_ids,
            f"{genotype_id} organism ID not in session organisms: {organism_id}",
        )


def validate_uniprot_ids(genes):
    uniprot_pattern = re.compile(
        r'^[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$'
    )
    for gene in genes.values():
        uniprot_id = gene['uniquename']
        _validate(
            uniprot_pattern.match(uniprot_id),
            f"invalid format for UniProtKB accession number: {uniprot_id}",
        )


def validate_unique_alleles(alleles):
    # Each allele in a session must be unique (duplicate allele objects are invalid).
    allele_keys = defaultdict(list)
    required_keys = ['gene', 'name', 'allele_type']
    optional_keys = ['description']
    for allele_id, allele in alleles.items():
        key_parts = [allele[k] for k in required_keys]
        for k in optional_keys:
            v = allele.get(k)
            if v:
                key_parts.append(v)
        allele_key = ' '.join(key_parts)
        _validate(
            allele_key not in allele_keys,
            "alleles are duplicates:\n{}".format(
                '\n'.join([*allele_keys[allele_key], allele_id])
            ),
        )
        allele_keys[allele_key].append(allele_id)


def validate_unique_annotations(annotations, session_id):
    seen = {}
    for annotation in annotations:
        if 'phi4_id' in annotation:
            key = str({k: v for k, v in annotation.items() if k != 'phi4_id'})
        else:
            key = str(annotation)
        dupe_annotation = seen.get(key)
        _validate(
            not dupe_annotation,
            (
                f"duplicate annotations: {session_id}\n"
                f"{pprint.pformat(annotation, indent=4)}\n"
                f"{pprint.pformat(dupe_annotation, indent=4)}\n"
            ),
        )
        seen[key] = annotation


def validate_unique_metagenotypes(metagenotypes):
    unique_metagenotypes = {}
    for metagenotype_id, metagenotype in metagenotypes.items():
        unique_id = ' '.join(
            (metagenotype['pathogen_genotype'], metagenotype['host_genotype'])
        )
        _validate(
            unique_id not in unique_metagenotypes,
            f"{metagenotype_id} is duplicate of {unique_id}",
        )
        unique_metagenotypes[unique_id] = metagenotype_id


def validate_phi4_ids_length(annotations):
    for annotation in annotations:
        phi4_ids = annotation.get('phi4_ids', [])
        if phi4_ids:
            phi4_ids_length = len('|'.join(phi4_ids))
            _validate(
                phi4_ids_length > 255,
                f"PHI-base 4 IDs are too long: {phi4_ids}"
            )


def validate_export(json_export, validate_id_length=False):
    schema = loaders.load_phibase5_json_schema()
    jsonschema.validate(json_export, schema)
    curation_sessions = json_export['curation_sessions']
    for session_id, session in curation_sessions.items():
        # Skip sessions that were not valid for curation
        if 'no_annotation_reason' in session['metadata']:
            continue
        # Consider removing this check once approved sessions are exported
        if session['metadata']['annotation_status'] != 'APPROVED':
            continue

        annotations = session['annotations']
        genes = session['genes']
        alleles = session['alleles']
        genotypes = session['genotypes']
        mutant_genotypes = {k: genotypes[k] for k in genotypes if 'wild-type' not in k}
        organisms = session['organisms']
        metagenotypes = session.get('metagenotypes', {})

        # Gene validation
        validate_uniprot_ids(genes)
        validate_gene_ids(genes)
        validate_gene_references(genes, organisms)
        # Organism validation
        validate_classified_organisms(organisms)
        # Allele validation
        validate_allele_ids(alleles)
        validate_allele_primary_ids(alleles)
        validate_allele_session_ids(alleles, session_id)
        validate_referenced_genes(alleles, genes)
        validate_unique_alleles(alleles)
        # Genotype validation
        validate_genotype_session_ids(mutant_genotypes, session_id)
        validate_referenced_alleles(genotypes, alleles)
        validate_referenced_organisms(genotypes, organisms)
        validate_one_species_per_genotype(genotypes, session)
        # Metagenotype validation
        validate_metagenotype_session_ids(metagenotypes, session_id)
        validate_referenced_genotypes(metagenotypes, genotypes)
        validate_unique_metagenotypes(metagenotypes)
        # Annotation validation
        validate_disease_ids(annotations)
        validate_unique_annotations(annotations, session_id)
        if validate_id_length:
            validate_phi4_ids_length(annotations)


validate_allele_session_ids = functools.partial(
    validate_feature_session_ids,
    'allele',  # feature type
    r'^[A-Z0-9]+:([a-f0-9]+)-\d+$',  # pattern
)
validate_genotype_session_ids = functools.partial(
    validate_feature_session_ids,
    'genotype',  # feature type
    r'([a-f0-9]+)-genotype-\d+',  # pattern
)
validate_metagenotype_session_ids = functools.partial(
    validate_feature_session_ids,
    'metagenotype',  # feature type
    r'([a-f0-9]+)-metagenotype-\d+',  # pattern
)
