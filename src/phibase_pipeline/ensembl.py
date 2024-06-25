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
