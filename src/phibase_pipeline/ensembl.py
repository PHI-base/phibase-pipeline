import json

import pandas as pd


def get_genotype_data(session, genotype_id, suffix='_a'):
    genotype = session['genotypes'][genotype_id]
    if len(genotype['loci']) > 1:
        return None
    alleles = session['alleles']
    genes = session['genes']
    strain = genotype['organism_strain']
    taxon_id = str(genotype['organism_taxonid'])
    species_name = session['organisms'][taxon_id]['full_name']
    uniprot_id = None
    allele_display_name = None
    for locus in genotype['loci']:
        for locus_allele in locus:
            expression = locus_allele.get('expression')
            expr_str = f' [{expression}]' if expression else ''
            allele_id = locus_allele['id']
            allele = alleles[allele_id]
            # Note the deliberate leading space!
            description = allele.get('description')
            desc_str = f'({description})' if description else ''
            allele_type = allele['allele_type'].replace('_', ' ')
            allele_display_name = (
                f"{allele['name']}{desc_str} ({allele_type}){expr_str}"
            )
            gene_id = allele['gene']
            gene = genes[gene_id]
            uniprot_id = gene['uniquename']
    record = {
        'uniprot': uniprot_id,
        'organism': species_name,
        'strain': strain,
        'modification': allele_display_name,
    }
    suffixed = {k + suffix: v for k, v in record.items()}
    return suffixed


def get_metagenotype_data(session, metagenotype_id):
    metagenotype = session['metagenotypes'][metagenotype_id]
    pathogen_genotype_id = metagenotype['pathogen_genotype']
    host_genotype_id = metagenotype['host_genotype']
    pathogen_columns = get_genotype_data(
        session, pathogen_genotype_id, suffix='_a'
    )
    if pathogen_columns is None:
        return None
    host_columns = get_genotype_data(
        session, host_genotype_id, suffix='_b'
    )
    if host_columns is None:
        return None
    return {**pathogen_columns, **host_columns}


def get_canto_columns(canto_export: dict) -> pd.DataFrame:

    def get_physical_interaction_columns(session, gene_id, interacting_id):
        raise NotImplementedError

    def get_tissue_ids(annotation):
        tissue_ids = [
            ext['rangeValue']
            for ext in annotation.get('extension', [])
            if ext['relation'] == 'infects_tissue'
        ]
        tissue_id_str = '; '.join(tissue_ids)
        return tissue_id_str or None

    column_order = [
        'uniprot_a',
        'organism_a',
        'strain_a',
        'modification_a',
        'uniprot_b',
        'organism_b',
        'strain_b',
        'modification_b',
        'phenotype',
        'disease',
        'host_tissue',
        'evidence_code',
        'interaction_type',
        'pmid',
    ]
    interaction_type_map = {
        'disease_name': 'interspecies interaction',
        'gene_for_gene_phenotype': 'gene-for-gene interaction',
        'pathogen_host_interaction_phenotype': 'interspecies interaction',
        'physical_interaction': 'protein-protein interaction',
    }
    curation_sessions = canto_export['curation_sessions']
    records = []
    for session in curation_sessions.values():
        pmid = int(next(iter(session['publications'].keys())).replace('PMID:', ''))
        for annotation in session.get('annotations', []):
            if metagenotype_id := annotation.get('metagenotype'):
                data = get_metagenotype_data(session, metagenotype_id)
            elif genotype_id := annotation.get('genotype'):
                # not loading single species phenotypes yet
                # data = get_genotype_data(session, genotype_id)
                continue
            elif annotation['type'] == 'physical_interaction':
                interaction_type = 'protein-protein interaction'
                data = get_physical_interaction_columns(
                    session,
                    gene_id=annotation['gene'],
                    interacting_id=None,
                )

            if data is None:
                continue
            interaction_type = interaction_type_map.get(annotation['type'])
            is_disease = annotation['type'] == 'disease_name'
            term_id = annotation['term']
            phenotype = None if is_disease else term_id
            disease = term_id if is_disease else None
            tissue_ids = get_tissue_ids(annotation)
            records.append(
                {
                    **data,
                    'phenotype': phenotype,
                    'disease': disease,
                    'host_tissue': tissue_ids,
                    'evidence_code': annotation['evidence_code'] or None,
                    'interaction_type': interaction_type,
                    'pmid': pmid,
                }
            )
    return pd.DataFrame.from_records(records)
