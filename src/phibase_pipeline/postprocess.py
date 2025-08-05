# SPDX-FileCopyrightText: 2023-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT

import copy
import re
from pathlib import Path

from phibase_pipeline import loaders, pubmed, uniprot


DATA_DIR = Path(__file__).parent / 'data'


def allele_ids_of_genotype(genotype):
    for locus in genotype['loci']:
        for locus_allele in locus:
            allele_id = locus_allele['id']
            yield allele_id


def merge_phi_canto_curation(phi_base_export, phi_canto_export):
    curation_sessions = phi_base_export['curation_sessions']
    old_session_count = len(curation_sessions)

    for session_id, session in phi_canto_export['curation_sessions'].items():
        if session_id in curation_sessions:
            raise KeyError(f"session_id {session_id} already exists in canto_json")
        curation_sessions[session_id] = session

    assert len(curation_sessions) > old_session_count


def merge_duplicate_alleles(curation_sessions):
    def get_allele_key(allele):
        required_keys = ('gene', 'allele_type')
        optional_keys = ('name', 'description')
        key_parts = [allele[k] for k in required_keys]
        for k in optional_keys:
            v = allele.get(k)
            if v:
                key_parts.append(v)
        return tuple(key_parts)

    def merge_synonyms(source_allele, target_allele):
        source_synonyms = source_allele['synonyms']
        target_synonyms = target_allele['synonyms']
        for synonym in source_synonyms:
            if synonym not in target_synonyms:
                target_synonyms.append(synonym)

    def update_references_to_alleles(allele_mapping, genotypes):
        for genotype in genotypes.values():
            for locus in genotype['loci']:
                for locus_allele in locus:
                    allele_id = locus_allele['id']
                    replacement_allele_id = allele_mapping.get(allele_id)
                    if replacement_allele_id:
                        locus_allele['id'] = replacement_allele_id

    for session in curation_sessions.values():
        allele_merge_mapping = {}
        allele_id_lookup = {}
        alleles = session.get('alleles', [])
        if len(alleles) < 2:
            # No duplicates unless 2 or more alleles
            continue
        for source_allele_id, source_allele in alleles.items():
            source_allele_key = get_allele_key(source_allele)
            target_allele_id = allele_id_lookup.get(source_allele_key)
            if target_allele_id:
                # Current allele is a duplicate; mark it for merging
                # and merge its synonyms with its duplicate allele.
                target_allele = alleles[target_allele_id]
                allele_merge_mapping[source_allele_id] = target_allele_id
                if source_allele['synonyms']:
                    merge_synonyms(source_allele, target_allele)
            else:
                allele_id_lookup[source_allele_key] = source_allele_id

        update_references_to_alleles(allele_merge_mapping, session['genotypes'])
        for merged_allele_id in allele_merge_mapping:
            del session['alleles'][merged_allele_id]


def remove_curator_orcids(export):
    for session in export['curation_sessions'].values():
        if 'annotations' not in session:
            continue
        for annotation in session['annotations']:
            if 'curator_orcid' in annotation['curator']:
                del annotation['curator']['curator_orcid']


def remove_invalid_annotations(session):
    id_feature_map = (
        ('gene', 'genes'),
        ('genotype', 'genotypes'),
        ('metagenotype', 'metagenotypes'),
    )
    valid_indexes = []
    annotations = session['annotations']
    for i, annotation in enumerate(annotations):
        for feature_key, feature_collection_key in id_feature_map:
            feature_id = annotation.get(feature_key)
            if feature_id is None:
                continue
            features = session.get(feature_collection_key, [])
            if feature_id in features:
                valid_indexes.append(i)
                break
    session['annotations'] = [annotations[i] for i in valid_indexes]


def remove_invalid_genotypes(session):
    def get_invalid_genotype_ids(session):
        invalid_ids = []
        genotypes = session['genotypes']
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
                    if taxon_id != genotype_taxon_id:
                        invalid_ids.append(genotype_id)
        return set(invalid_ids)

    invalid_ids = get_invalid_genotype_ids(session)
    genotypes = session['genotypes']
    for genotype_id in invalid_ids:
        del genotypes[genotype_id]


def remove_invalid_metagenotypes(session):
    def get_invalid_metagenotype_ids(genotypes, metagenotypes):
        return [
            metagenotype_id
            for metagenotype_id, metagenotype in metagenotypes.items()
            if metagenotype['pathogen_genotype'] not in genotypes
        ]

    genotypes = session['genotypes']
    metagenotypes = session['metagenotypes']
    for metagenotype_id in get_invalid_metagenotype_ids(genotypes, metagenotypes):
        del session['metagenotypes'][metagenotype_id]


def remove_orphaned_alleles(session):
    def get_orphaned_alleles(genotypes, alleles):
        return [
            allele_id
            for allele_id in alleles
            if not any(
                allele_id == genotype_allele_id
                for genotype in genotypes.values()
                for genotype_allele_id in allele_ids_of_genotype(genotype)
            )
        ]

    alleles = session['alleles']
    genotypes = session['genotypes']
    for allele_id in get_orphaned_alleles(genotypes, alleles):
        del session['alleles'][allele_id]


def remove_orphaned_genes(session):
    orphaned_gene_ids = set()
    for gene_id, gene in session['genes'].items():
        # Check alleles
        for allele in session['alleles'].values():
            if allele['gene'] == gene_id:
                break
        else:  # Check annotations
            annotations = session['annotations']
            for annotation in annotations:
                annotation_gene_id = annotation.get('gene')
                if annotation_gene_id == gene_id:
                    break
                # Check annotation extensions
                uniprot_id = gene['uniquename']
                gene_in_extensions = any(
                    extension['rangeValue'] == uniprot_id
                    for extension in annotation['extension']
                )
                if gene_in_extensions:
                    break
            else:
                orphaned_gene_ids.add(gene_id)
    for gene_id in orphaned_gene_ids:
        del session['genes'][gene_id]


def remove_orphaned_organisms(session):
    orphaned_organism_ids = set()
    for organism_id, organism in session['organisms'].items():
        organism_name = organism['full_name']
        # Check genes
        for gene in session['genes'].values():
            if gene['organism'] == organism_name:
                break
        else:  # Check genotypes
            taxon_id = int(organism_id)
            for genotype in session['genotypes'].values():
                if genotype['organism_taxonid'] == taxon_id:
                    break
            else:
                orphaned_organism_ids.add(organism_id)
    for organism_id in orphaned_organism_ids:
        del session['organisms'][organism_id]


def remove_duplicate_annotations(phibase_json):
    seen = set()
    curation_sessions = phibase_json['curation_sessions']
    for session in curation_sessions.values():
        unique_annotations = []
        annotations = session.get('annotations', [])
        for annotation in annotations:
            key = (
                str({k: v for k, v in annotation.items() if k != 'phi4_id'})
                if 'phi4_id' in annotation
                else str(annotation)
            )
            if key in seen:
                continue  # skip duplicate annotation
            unique_annotations.append(annotation)
            seen.add(key)
        session['annotations'] = unique_annotations


def remove_allele_gene_names(session):
    for allele in session.get('alleles', {}).values():
        if 'gene_name' in allele:
            del allele['gene_name']


def add_delta_symbol(session):
    """Replace 'delta' with the capital delta symbol in allele names."""
    pattern = re.compile(r'delta')
    DELTA = '\N{GREEK CAPITAL LETTER DELTA}'
    for allele in session.get('alleles', {}).values():
        allele['name'] = pattern.sub(DELTA, allele['name'])


def add_chemical_extensions(export, chemical_data):
    extensions_and_relations = [
        ('chebi_id', 'chebi_id'),
        ('frac', 'frac_code'),
        ('cas', 'cas_number'),
        ('smiles', 'smiles'),
    ]
    for session in export['curation_sessions'].values():
        for annotation in session.get('annotations', []):
            extensions = annotation.get('extension')
            if extensions is None:
                continue
            term_data = chemical_data.get(annotation.get('term'))
            if not term_data:
                continue
            for ext_name, relation in extensions_and_relations:
                range_value = term_data[ext_name]
                if range_value:
                    extensions.append(
                        {
                            'rangeDisplayName': '',
                            'rangeType': 'Text',
                            'rangeValue': range_value,
                            'relation': relation,
                        }
                    )


def remove_unapproved_sessions(export):
    curation_sessions = export['curation_sessions']
    # Copy dict keys as list since we'll be deleting dict keys
    for session_id in list(curation_sessions):
        session = curation_sessions[session_id]
        if session['metadata']['annotation_status'] != 'APPROVED':
            del curation_sessions[session_id]


def get_all_uniprot_ids_in_export(export):
    return set(
        gene['uniquename']
        for session in export['curation_sessions'].values()
        if 'genes' in session
        for gene in session['genes'].values()
    )


def add_uniprot_data_to_genes(export, uniprot_gene_data):
    augmented_export = copy.deepcopy(export)
    for session in augmented_export['curation_sessions'].values():
        genes = session.get('genes', {})
        for gene in genes.values():
            uniprot_id = gene['uniquename']
            uniprot_data = uniprot_gene_data.get(uniprot_id)
            if uniprot_data is None:
                print(f'warning: {uniprot_id} not found in UniProt data')
                continue
            gene['uniprot_data'] = uniprot_data
    return augmented_export


def add_proteome_strains_to_genes(export, proteome_results, proteome_id_mapping):
    proteome_strain_mapping = {
        proteome['id']: proteome.get('strain') for proteome in proteome_results['results']
    }
    # Link to the strain of the first listed proteome only
    uniprot_strain_mapping = {
        uniprot_id: proteome_strain_mapping[proteome_ids[0]] if proteome_ids else None
        for uniprot_id, proteome_ids in proteome_id_mapping.items()
    }
    augmented_export = copy.deepcopy(export)
    for session in augmented_export['curation_sessions'].values():
        genes = session.get('genes', {})
        for gene in genes.values():
            uniprot_id = gene['uniquename']
            # Use False as the default value because None is already
            # being used to indicate missing strain names
            strain = uniprot_strain_mapping.get(uniprot_id, False)
            if strain is False:
                continue
            gene['uniprot_data']['strain'] = strain
    return augmented_export


def get_all_pmids_in_export(export):
    return set(
        pmid.replace('PMID:', '')
        for session in export['curation_sessions'].values()
        for pmid in session['publications']
    )


def add_pubmed_data_to_sessions(export, pubmed_data):
    augmented_export = copy.deepcopy(export)
    for session in augmented_export['curation_sessions'].values():
        publications = session['publications']
        pmid = next(iter(publications.keys()))
        publication_data = pubmed_data.get(pmid)
        if publication_data is None:
            print(f'session PMID not found in PubMed data: {pmid}')
            continue
        publications[pmid]['pubmed_data'] = publication_data
    return augmented_export


def add_cross_references(export):
    session = uniprot.make_session()
    # UniProt data
    uniprot_ids = get_all_uniprot_ids_in_export(export)
    id_mapping_results = uniprot.run_id_mapping_job(session, uniprot_ids)
    uniprot_gene_data = uniprot.get_uniprot_data_fields(id_mapping_results)
    export = add_uniprot_data_to_genes(export, uniprot_gene_data)
    # Sequence strains
    proteome_id_mapping = uniprot.get_proteome_id_mapping(id_mapping_results)
    proteome_results = uniprot.query_proteome_ids(session, proteome_id_mapping)
    export = add_proteome_strains_to_genes(export, proteome_results, proteome_id_mapping)
    # Publication information
    pmids = get_all_pmids_in_export(export)
    pubmed_data = pubmed.get_publications_from_pubmed(pmids)
    publication_fields = pubmed.get_all_publication_details(pubmed_data)
    export = add_pubmed_data_to_sessions(export, publication_fields)
    return export


def truncate_long_values(export):
    """Truncate list values that are too long to be loaded into the
    database used by the PHI-base 5 upload tool."""

    def truncate_list(lst, length, sep_length):
        # Assume ID lists shorter than this are always under
        # the limit. This number can probably be higher.
        if len(lst) < 2:
            return lst
        running_length = 0
        truncated_list = []
        for item in lst:
            running_length += len(item)
            if running_length > length:
                break
            # Add separator to the count after the length check, because
            # the concatenated list of IDs doesn't end with a separator.
            running_length += sep_length
            truncated_list.append(item)
        return truncated_list

    def truncate_ensembl_ids(export, length_limit, sep_length):
        keys = ('ensembl_gene_id', 'ensembl_sequence_id')
        for session in export['curation_sessions'].values():
            genes = session.get('genes', {}).values()
            for gene in genes:
                uniprot_data = gene['uniprot_data']
                for key in keys:
                    id_list = uniprot_data.get(key)
                    if id_list is None:
                        continue
                    truncated_ids = truncate_list(id_list, length_limit, sep_length)
                    gene['uniprot_data'][key] = truncated_ids

    def truncate_allele_descriptions(export, length_limit, sep_length):
        description_sep = '; '
        sep_length = len(description_sep)
        delta = '\N{GREEK CAPITAL LETTER DELTA}'
        for session in export['curation_sessions'].values():
            alleles = session.get('alleles', {}).values()
            for allele in alleles:
                description = allele.get('description', '')
                # The data loading script escapes non latin-1 characters as
                # HTML entities, so we need to account for this.
                escaped_description = description.replace(delta, '&Delta;')
                if len(escaped_description) <= length_limit:
                    continue
                description_list = escaped_description.split(description_sep)
                # Append an ellipsis to indicate truncation. Note we can't
                # use a Unicode ellipsis since it's not in latin-1.
                end_ellipsis = ' ...'
                length_limit = length_limit - len(end_ellipsis)
                truncated_description = description_sep.join(
                    truncate_list(description_list, length_limit, sep_length)
                )
                final_description = (
                    truncated_description.replace('&Delta;', delta) + end_ellipsis
                )
                allele['description'] = final_description

    def truncate_phi4_ids(export, length_limit, sep_length):
        for session in export['curation_sessions'].values():
            for annotation in session.get('annotations', []):
                phi4_ids = annotation.get('phi4_id')
                if phi4_ids:
                    annotation['phi4_id'] = truncate_list(
                        phi4_ids, length_limit, sep_length
                    )

    LENGTH_LIMIT = 255  # length limit in database
    SEP_LENGTH = len('|')  # separator used by data upload tool
    args = (export, LENGTH_LIMIT, SEP_LENGTH)
    truncate_ensembl_ids(*args)
    truncate_allele_descriptions(*args)
    truncate_phi4_ids(*args)


def replace_obsolete_phido_terms(export, obsolete_phido_mapping):
    for session in export['curation_sessions'].values():
        annotations = session.get('annotations', [])
        if not annotations:
            continue
        annotations_to_keep = []
        for annotation in annotations:
            if annotation['type'] != 'disease_name':
                annotations_to_keep.append(annotation)
                continue
            term_id = annotation['term']
            replacements = obsolete_phido_mapping.get(term_id)
            if replacements is None:
                # Do not assume unmapped terms are obsolete
                annotations_to_keep.append(annotation)
                continue
            num_replacements = len(replacements)
            if num_replacements == 0:
                # Term is obsolete with no replacement: skip it
                continue
            if num_replacements == 1:
                annotation['term'] = replacements[0]
                annotations_to_keep.append(annotation)
            elif num_replacements > 1:
                for replacement_term in replacements:
                    new_annotation = annotation.copy()
                    new_annotation['term'] = replacement_term
                    annotations_to_keep.append(new_annotation)
        session['annotations'] = annotations_to_keep


def postprocess_phibase_json(export):
    curation_sessions = export['curation_sessions']
    for session in curation_sessions.values():
        remove_invalid_genotypes(session)
        remove_invalid_metagenotypes(session)
        remove_invalid_annotations(session)
        remove_orphaned_alleles(session)
        remove_orphaned_genes(session)
        remove_orphaned_organisms(session)
    remove_duplicate_annotations(export)


def postprocess_combined_json(export):
    chemical_data = loaders.load_chemical_data()
    obsolete_phido_mapping = loaders.load_obsolete_phido_mapping()
    remove_unapproved_sessions(export)
    remove_curator_orcids(export)
    merge_duplicate_alleles(export['curation_sessions'])
    add_chemical_extensions(export, chemical_data)
    replace_obsolete_phido_terms(export, obsolete_phido_mapping)
    for session in export['curation_sessions'].values():
        remove_allele_gene_names(session)
        add_delta_symbol(session)
