import re


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

    def merge_synonyms(existing_allele, dupe_allele):
        existing_synonyms = existing_allele['synonyms']
        for synonym in dupe_allele['synonyms']:
            if synonym not in existing_synonyms:
                existing_synonyms.append(synonym)

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
        allele_keys = {}
        alleles = session.get('alleles', [])
        if len(alleles) < 2:
            # No duplicates unless 2 or more alleles
            continue
        for allele_id, allele in alleles.items():
            allele_key = get_allele_key(allele)
            existing_allele_id = allele_keys.get(allele_key)
            if existing_allele_id:
                # Current allele is a duplicate; mark it for merging
                # and merge its synonyms with its duplicate allele.
                existing_allele = alleles[existing_allele_id]
                allele_merge_mapping[allele_id] = existing_allele_id
                if allele['synonyms']:
                    merge_synonyms(existing_allele, allele)
            else:
                allele_keys[allele_key] = allele_id

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
    """ Replace 'delta' with the capital delta symbol in allele names. """
    pattern = re.compile(r'delta$')
    DELTA = '\N{GREEK CAPITAL LETTER DELTA}'
    for allele in session.get('alleles', {}).values():
        allele['name'] = pattern.sub(DELTA, allele['name'])


def remove_unapproved_sessions(export):
    curation_sessions = export['curation_sessions']
    # Copy dict keys as list since we'll be deleting dict keys
    for session_id in list(curation_sessions):
        session = curation_sessions[session_id]
        if session['metadata']['annotation_status'] != 'APPROVED':
            del curation_sessions[session_id]


def postprocess_phibase_json(export):
    curation_sessions = export['curation_sessions']
    for session in curation_sessions.values():
        remove_invalid_genotypes(session)
        remove_invalid_metagenotypes(session)
        remove_invalid_annotations(session)
        remove_orphaned_alleles(session)
        remove_orphaned_genes(session)
        remove_orphaned_organisms(session)
    merge_duplicate_alleles(curation_sessions)
    remove_duplicate_annotations(export)


def postprocess_combined_json(export):
    remove_unapproved_sessions(export)
    remove_curator_orcids(export)
    for session in export['curation_sessions'].values():
        remove_allele_gene_names(session)
        add_delta_symbol(session)
