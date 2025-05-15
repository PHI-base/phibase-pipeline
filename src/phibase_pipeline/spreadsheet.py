# SPDX-FileCopyrightText: 2023-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT


import json
import re
from collections import defaultdict

import pandas as pd


def add_high_level_terms(export):
    """Add PHI-base 4 high-level terms as properties of annotation objects."""

    def is_effector_genotype(genotype):
        for locus in genotype['loci']:
            for locus_allele in locus:
                allele_id = locus_allele['id']
                allele = session['alleles'][allele_id]
                gene_id = allele['gene']
                gene = session['genes'][gene_id]
                uniprot_id = gene['uniquename']
                if uniprot_id in effector_uniprot_ids:
                    return True
        return False

    term_mapping = {
        'GO:0140418': 'Effector',
        'PHIPO:0000022': 'Resistance to chemical',
        'PHIPO:0000021': 'Sensitivity to chemical',
        'PHIPO:0000513': 'Lethal',
    }
    extension_mapping = {
        "PHIPO:0000207": "Loss of mutualism",
        "PHIPO:0000014": "Increased virulence",
        "PHIPO:0000010": "Loss of pathogenicity",
        "PHIPO:0000015": "Reduced virulence",
        "PHIPO:0000004": "Unaffected pathogenicity",
    }
    effector_uniprot_ids = set()

    for session in export['curation_sessions'].values():
        annotations = session.get('annotations', [])
        for annotation in annotations:
            high_level_terms = set()
            term = annotation.get('term')
            if not term:
                annotation['high_level_terms'] = []
                continue
            high_level_term = term_mapping.get(term)
            if high_level_term:
                if high_level_term == 'Effector':
                    gene = session['genes'][annotation['gene']]
                    uniprot_id = gene['uniquename']
                    effector_uniprot_ids.add(uniprot_id)
                high_level_terms.add(high_level_term)
            extensions = annotation.get('extension')
            if extensions:
                for extension in extensions:
                    extension_term = extension['rangeValue']
                    high_level_term = extension_mapping.get(extension_term)
                    if high_level_term:
                        high_level_terms.add(high_level_term)
            # Add Effector terms for features containing effector genes
            genotype = None
            if 'genotype' in annotation:
                genotype_id = annotation['genotype']
                genotype = session['genotypes'][genotype_id]
            elif 'metagenotype' in annotation:
                metagenotype_id = annotation['metagenotype']
                metagenotype = session['metagenotypes'][metagenotype_id]
                pathogen_genotype_id = metagenotype['pathogen_genotype']
                genotype = session['genotypes'][pathogen_genotype_id]
            if genotype and is_effector_genotype(genotype):
                high_level_terms.add('Effector')
            annotation['high_level_terms'] = sorted(list(high_level_terms))


def allele_has_gene(gene_id, allele):
    return allele['gene'] == gene_id


def genotype_has_gene(gene_id, session, genotype):
    alleles = session['alleles']
    for locus in genotype['loci']:
        for locus_allele in locus:
            allele_id = locus_allele['id']
            allele = alleles[allele_id]
            if allele_has_gene(gene_id, allele):
                return True
    return False


def metagenotype_has_gene(gene_id, session, metagenotype):
    for key in ('pathogen_genotype', 'host_genotype'):
        genotype_id = metagenotype[key]
        genotype = session['genotypes'][genotype_id]
        if genotype_has_gene(gene_id, session, genotype):
            return True
    return False


def annotation_has_gene(gene_id, annotation, session):
    metagenotype_id = annotation.get('metagenotype')
    if metagenotype_id:
        metagenotype = session['metagenotypes'][metagenotype_id]
        return metagenotype_has_gene(gene_id, session, metagenotype)
    genotype_id = annotation.get('genotype')
    if genotype_id:
        genotype = session['genotypes'][genotype_id]
        return genotype_has_gene(gene_id, session, genotype)
    annotation_gene = annotation.get('gene')
    if gene_id:
        return annotation_gene == gene_id
    return False


def get_gene_high_level_terms(gene_id, session):
    return set(
        term
        for annotation in session['annotations']
        for term in annotation['high_level_terms']
        if annotation_has_gene(gene_id, annotation, session)
    )


def get_entry_summary_table(export):
    maybe_join = lambda x: x if x is None else ('; '.join(x) or None)
    records = []
    for session in export['curation_sessions'].values():
        genes = session.get('genes')
        if not genes:
            continue
        for gene_id, gene in genes.items():
            uniprot_id = gene['uniquename']
            phig_id = gene['phig_id']
            uniprot_data = gene.get('uniprot_data')
            if not uniprot_data:
                continue  # Assume accession is obsolete
            gene_name = uniprot_data['name']
            high_level_terms = get_gene_high_level_terms(gene_id, session)
            records.append(
                {
                    'phig_id': phig_id,
                    'gene_name': gene_name,
                    'species': gene['organism'],
                    'uniprot_id': uniprot_id,
                    'gene_product': uniprot_data['product'],
                    'sequence_strain': uniprot_data['strain'],
                    'ensembl_genomes': maybe_join(uniprot_data['ensembl_sequence_id']),
                    'ncbi_genbank': uniprot_data['dbref_gene_id'],
                    'high_level_term': maybe_join(high_level_terms),
                }
            )
    return pd.DataFrame.from_records(records)


def get_annotation_extensions(annotation):
    extensions = annotation.get('extension')
    if not extensions:
        return None
    text_parts = []
    for ext in extensions:
        relation = ext['relation']
        display_name = ext.get('rangeDisplayName')
        ext_value = ext['rangeValue']
        ext_text = f'{relation} {display_name} ({ext_value})'
        text_parts.append(ext_text)
    extension_text = '; '.join(text_parts)
    return extension_text or None


def get_experimental_conditions(annotation, term_label_mapping):
    conditions = annotation.get('conditions')
    if not conditions:
        return None
    condition_text = '; '.join(
        f'{term_label_mapping.get(term_id, "NO_LABEL")} ({term_id})'
        for term_id in conditions
    )
    return condition_text


def get_ontology_annotation_tables(export, term_label_mapping):
    annotation_types = defaultdict(list)
    for session in export['curation_sessions'].values():
        for annotation in session['annotations']:
            annotation_type = annotation['type']
            if annotation_type == 'physical_interaction':
                continue  # Will be handled by the get_physical_interaction_table function
            extensions = get_annotation_extensions(annotation)
            conditions = get_experimental_conditions(annotation, term_label_mapping)
            feature_key = next(
                k for k in ('gene', 'genotype', 'metagenotype') if k in annotation
            )
            term_id = annotation['term']
            annotation_types[annotation_type].append(
                {
                    'term_id': term_id,
                    'term_label': term_label_mapping[term_id],
                    feature_key: annotation[feature_key],
                    'annotation_extensions': extensions,
                    'evidence': annotation.get('evidence_code'),
                    'conditions': conditions,
                    'high_level_term': '; '.join(annotation['high_level_terms']) or None,
                    'phi4_id': '; '.join(annotation.get('phi4_id', [])) or None,
                    'publication': annotation['publication'],
                }
            )
    dfs = {
        annotation_type: pd.DataFrame.from_records(annotations)
        for annotation_type, annotations in annotation_types.items()
    }
    return dfs


def get_physical_interaction_table(export):
    records = []
    for session in export['curation_sessions'].values():
        annotations = [
            annotation
            for annotation in session['annotations']
            if annotation['type'] == 'physical_interaction'
        ]
        for annotation in annotations:
            records.append(
                {
                    'gene': annotation['gene'],
                    'interacting_gene': annotation['interacting_genes'][0],
                    'evidence': annotation.get('evidence_code'),
                    'phi4_id': '; '.join(annotation.get('phi4_id', [])) or None,
                    'publication': annotation['publication'],
                }
            )
    return pd.DataFrame.from_records(records)


def get_publication_table(export):
    records = []
    for session in export['curation_sessions'].values():
        for pmid, publication in session['publications'].items():
            pubmed_data = publication.get('pubmed_data')
            if not pubmed_data:
                continue
            name = pubmed_data['author']
            year = pubmed_data['year']
            reference = f'{name} et al., {year}'
            citation_parts = ['{journal_abbr}. {year}; {volume}']
            if pubmed_data['issue']:
                citation_parts.append('({issue})')
            if pubmed_data['pages']:
                citation_parts.append(': {pages}')
            citation_template = ''.join(citation_parts)
            citation = citation_template.format(**pubmed_data)
            records.append(
                {
                    'pmid': pmid,
                    'reference': reference,
                    'journal_citation': citation,
                    'year': int(year),
                }
            )
    return pd.DataFrame.from_records(records)


def get_strain_table(export):
    gene_strains = defaultdict(set)
    for session in export['curation_sessions'].values():
        organisms = session['organisms']
        for gene_id, gene in session.get('genes', {}).items():
            uniprot_id = gene['uniquename']
            taxid = get_taxid_of_gene(gene, organisms)
            organism = organisms[str(taxid)]
            species = organism['full_name']
            role = organism['role']
            key = (taxid, species, role, uniprot_id)
            strains = set(
                genotype.get('organism_strain')
                for genotype in session['genotypes'].values()
                if genotype_has_gene(gene_id, session, genotype)
            )
            gene_strains[key].update(strains)
    records = []
    for key, strains in gene_strains.items():
        taxid, species, role, uniprot_id = key
        filtered_strains = [s for s in strains if s is not None]
        for strain in filtered_strains:
            records.append(
                {
                    'taxon_id': taxid,
                    'species': species,
                    'role': role,
                    'uniprot_id': uniprot_id,
                    'strain': strain,
                }
            )
    return pd.DataFrame.from_records(records)


def get_taxid_of_gene(gene, organisms):
    organism_name = gene['organism']
    for taxid, organism in organisms.items():
        if organism['full_name'] == organism_name:
            return int(taxid)
    raise ValueError(f"organism {organism_name} not found in organisms:\n{organisms}")


def iter_genes_of_genotype(session, genotype):
    for locus in genotype['loci']:
        for locus_allele in locus:
            allele_id = locus_allele['id']
            allele = session['alleles'][allele_id]
            gene_id = allele['gene']
            gene = session['genes'][gene_id]
            yield (gene_id, gene)


def iter_genes_of_metagenotype(session, metagenotype):
    for role in ('pathogen', 'host'):
        genotype_key = role + '_genotype'
        genotype_id = metagenotype[genotype_key]
        genotype = session['genotypes'][genotype_id]
        for gene_id, gene in iter_genes_of_genotype(session, genotype):
            yield (gene_id, gene, role)


def get_interactions_table(export):
    interacting_records = []
    curation_sessions = export['curation_sessions']
    for session in curation_sessions.values():
        genes = session.get('genes')
        if not genes:
            continue
        genotypes = session['genotypes']
        metagenotypes = session.get('metagenotypes', {})
        organisms = session['organisms']
        for gene_id_a, gene_a in genes.items():
            uniprot_a = gene_a['uniquename']
            taxid_a = get_taxid_of_gene(gene_a, organisms)
            organism_a = organisms[str(taxid_a)]
            species_a = organism_a['full_name']
            role_a = organism_a['role']
            role_b = 'host' if role_a == 'pathogen' else 'pathogen'
            genotype_key = f'{role_b}_genotype'
            # Pathogen-host interactions
            for metagenotype in metagenotypes.values():
                genotype_id = metagenotype[genotype_key]
                genotype = genotypes[genotype_id]
                taxid_b = genotype['organism_taxonid']
                species_b = organisms[str(taxid_b)]['full_name']
                interacting_genes = [
                    gene_id for gene_id, _ in iter_genes_of_genotype(session, genotype)
                ]
                if not interacting_genes:
                    interacting_genes = [None]
                for gene_id_b in interacting_genes:
                    gene_b = genes.get(gene_id_b)
                    uniprot_b = gene_b['uniquename'] if gene_b else None
                    interacting_records.append(
                        {
                            'taxid_a': taxid_a,
                            'species_a': species_a,
                            'uniprot_a': uniprot_a,
                            'taxid_b': taxid_b,
                            'species_b': species_b,
                            'uniprot_b': uniprot_b,
                        }
                    )
            # Physical interactions
            for annotation in session['annotations']:
                if annotation['type'] == 'physical_interaction':
                    gene_id_b = annotation['interacting_genes'][0]
                    if annotation['gene'] == gene_id_a:
                        gene_b = session['genes'][gene_id_b]
                    elif gene_id_b == gene_id_a:
                        gene_b = session['genes'][annotation['gene']]
                    else:
                        continue
                    taxid_b = get_taxid_of_gene(gene_b, organisms)
                    species_a, uniprot_a = gene_id_a.rsplit(' ', maxsplit=1)
                    species_b, uniprot_b = gene_id_b.rsplit(' ', maxsplit=1)
                    interacting_records.append(
                        {
                            'taxid_a': taxid_a,
                            'species_a': uniprot_a,
                            'uniprot_a': gene_id_a,
                            'taxid_b': taxid_b,
                            'species_b': gene_id_b,
                            'uniprot_b': uniprot_b,
                        }
                    )
    return pd.DataFrame.from_records(interacting_records).drop_duplicates()


def get_display_name_lookup(export):

    def get_locus_allele_display_name(session, locus_allele):
        allele_id = locus_allele['id']
        allele = session['alleles'][allele_id]
        name = allele.get('name', 'unnamed')
        encoded_name = re.sub('delta$', '\N{GREEK CAPITAL LETTER DELTA}', name)
        allele_type = allele['allele_type']
        description = allele.get('description', '')
        if description:
            description = f'({description})'
        expression = locus_allele.get('expression', '')
        if expression:
            expression = f'[{expression}]'
        allele_display_name = f'{encoded_name}{description} ({allele_type}){expression}'
        return allele_display_name

    def get_genotype_display_name(session, genotype):
        taxon_id = genotype['organism_taxonid']
        organism = session['organisms'][str(taxon_id)]
        organism_name = organism['full_name']
        genus, rest = organism_name.split(maxsplit=1)
        scientific_name = f'{genus} {rest}'
        strain = genotype.get('organism_strain', 'Unknown strain')
        loci = genotype['loci']
        if not loci:
            return f'wild type {scientific_name} ({strain})'
        parts = []
        for locus in loci:
            for locus_allele in locus:
                locus_name = get_locus_allele_display_name(session, locus_allele)
                parts.append(f'{locus_name} {scientific_name} ({strain})')
        return ' '.join(parts)

    def get_metagenotype_display_name(session, metagenotype):
        genotype_display_names = []
        for genotype_id in (
            metagenotype['pathogen_genotype'],
            metagenotype['host_genotype'],
        ):
            genotype = session['genotypes'][genotype_id]
            genotype_display_names.append(get_genotype_display_name(session, genotype))
        return ' '.join(genotype_display_names)

    lookup = {}
    feature_functions = {
        'genotypes': get_genotype_display_name,
        'metagenotypes': get_metagenotype_display_name,
    }
    for session in export['curation_sessions'].values():
        for feature_key, display_func in feature_functions.items():
            features = session.get(feature_key)
            if features is None:
                break  # Every feature depends on the previous type
            for feature_id, feature in features.items():
                lookup[feature_id] = display_func(session, feature)
    return lookup


def add_display_names(dfs, display_name_lookup):
    for df in dfs.values():
        for feature_key in ('genotype', 'metagenotype'):
            if feature_key in df:
                loc = df.columns.get_loc(feature_key) + 1
                column_name = feature_key + '_name'
                values = df[feature_key].map(display_name_lookup)
                df.insert(loc, column_name, values)
                break


def get_genotype_uniprot_column(export, genotype_ids):
    curation_sessions = export['curation_sessions']
    rows = []
    for gid in genotype_ids.values:
        session_id, feature_type, *_ = gid.split('-')
        if feature_type != 'genotype':
            raise ValueError(f'invalid feature ID: {gid}')
        session = curation_sessions[session_id]
        genotype = session['genotypes'][gid]
        genes = iter_genes_of_genotype(session, genotype)
        row = '; '.join(gene['uniquename'] for _, gene in genes)
        rows.append(row)
    return pd.Series(rows, index=genotype_ids.index, name='uniprot_id')


def get_metagenotype_uniprot_column(export, metagenotype_ids):
    curation_sessions = export['curation_sessions']
    pathogen_rows = []
    host_rows = []
    for mid in metagenotype_ids.values:
        pathogen_uniprot_ids = []
        host_uniprot_ids = []
        session_id, feature_type, *_ = mid.split('-')
        if feature_type != 'metagenotype':
            raise ValueError(f'invalid feature ID: {mid}')
        session = curation_sessions[session_id]
        metagenotype = session['metagenotypes'][mid]
        genes = iter_genes_of_metagenotype(session, metagenotype)
        for _, gene, role in genes:
            uniprot_id = gene['uniquename']
            if role == 'pathogen':
                pathogen_uniprot_ids.append(uniprot_id)
            elif role == 'host':
                host_uniprot_ids.append(uniprot_id)
            else:
                raise ValueError(f'invalid role: {gene}')
        pathogen_rows.append('; '.join(pathogen_uniprot_ids))
        host_rows.append('; '.join(host_uniprot_ids))
    index = metagenotype_ids.index
    return (
        pd.Series(pathogen_rows, index, name='pathogen_uniprot'),
        pd.Series(host_rows, index, name='host_uniprot'),
    )


def add_feature_uniprot_columns(export, df):
    if 'genotype' in df.columns:
        df = df.copy()
        loc = df.columns.get_loc('genotype')
        values = get_genotype_uniprot_column(export, df['genotype'])
        df.insert(loc, 'uniprot_id', values)
    elif 'metagenotype' in df.columns:
        df = df.copy()
        loc = df.columns.get_loc('metagenotype')
        pathogen_col, host_col = get_metagenotype_uniprot_column(
            export, df['metagenotype']
        )
        # Order should be pathogen_uniprot, host_uniprot
        df.insert(loc, 'host_uniprot', host_col)
        df.insert(loc, 'pathogen_uniprot', pathogen_col)
    elif 'gene' in df.columns:
        df = df.copy()
        loc = df.columns.get_loc('gene')
        uniprot_ids = pd.Series(
            gene_id.rsplit(maxsplit=1)[1] for gene_id in df['gene'].values
        )
        column_name = 'uniprot_id'
        if 'interacting_gene' in df.columns:
            # Assume we're in the physical_interaction table (dangerous?)
            loc_b = df.columns.get_loc('interacting_gene')
            uniprot_ids_b = pd.Series(
                gene_id.rsplit(maxsplit=1)[1] for gene_id in df['interacting_gene'].values
            )
            df.insert(loc_b, 'uniprot_b', uniprot_ids_b)
            column_name = 'uniprot_a'
        df.insert(loc, column_name, uniprot_ids)
    return df  # Return the DataFrame unmodified if no cases match


def replace_merged_accessions(export, gene_data):
    export_text = json.dumps(export)
    id_mapping = {
        old_id: new_id
        for old_id, gene in gene_data.items()
        if old_id != (new_id := gene['uniprot_id'])
    }
    pattern = re.compile(f"({'|'.join(id_mapping.keys())})")
    export_text = pattern.sub(lambda m: id_mapping[m.group(0)], export_text)
    return json.loads(export_text)


def add_phig_ids(export, phig_lookup):
    for session in export['curation_sessions'].values():
        for gene in session.get('genes', {}).values():
            uniprot_id = gene['uniquename']
            gene['phig_id'] = phig_lookup.get(uniprot_id)


def make_spreadsheet_dataframes(
    export,
    gene_data,
    phig_mapping,
    term_label_mapping,
):
    export = replace_merged_accessions(export, gene_data)
    add_high_level_terms(export)
    add_phig_ids(export, phig_mapping)
    dfs = {
        'entry_summary': get_entry_summary_table(export),
        **get_ontology_annotation_tables(export, term_label_mapping),
        'physical_interaction': get_physical_interaction_table(export),
        'publication': get_publication_table(export),
        'strain': get_strain_table(export),
        'interaction': get_interactions_table(export),
    }
    display_name_lookup = get_display_name_lookup(export)
    add_display_names(dfs, display_name_lookup)
    for k, df in dfs.items():
        dfs[k] = add_feature_uniprot_columns(export, df)
    sheet_renames = {
        'biological_process': 'go_biological_process',
        'cellular_component': 'go_cellular_component',
        'disease_name': 'disease',
        'molecular_function': 'go_molecular_function',
        'pathogen_host_interaction_phenotype': 'phi_phenotype',
        'post_translational_modification': 'protein_modification',
        'wt_rna_expression': 'wild_type_rna_level',
        'wt_protein_expression': 'wild_type_protein_level',
    }
    for old_name, new_name in sheet_renames.items():
        dfs[new_name] = dfs.pop(old_name)
    sheet_order = [
        'entry_summary',
        'phi_phenotype',
        'gene_for_gene_phenotype',
        'pathogen_phenotype',
        'host_phenotype',
        'go_biological_process',
        'go_cellular_component',
        'go_molecular_function',
        'physical_interaction',
        'protein_modification',
        'disease',
        'wild_type_protein_level',
        'wild_type_rna_level',
        'strain',
        'interaction',
        'publication',
    ]
    spreadsheet_dfs = {k: dfs[k] for k in sheet_order}
    return spreadsheet_dfs


def make_spreadsheet_file(spreadsheet_dfs, output_path):
    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, df in spreadsheet_dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
