"""RCSB api utils."""

import warnings
from collections.abc import Iterable

import requests

QUERY_TEMPLATE = """
{
  nonpolymer_entities(entity_ids: $entity_ids) {
    rcsb_id
    nonpolymer_entity_instances {
      rcsb_nonpolymer_instance_validation_score {
        ranking_model_fit
      }
    }
  }
}
"""


def fetch_ranking_model_fit(entities: Iterable[str]) -> dict[str, float]:
    """Fetch ranking model fit scores from RCSB GraphQL API.

    Parameters
    ----------
    entities : list[str]
        List of non-polymer entity IDs in the format "{pdb_id}_{entity_id}".

    Returns
    -------
    dict[str, float]
        Mapping from entity ID to ranking model fit score.
    """
    url = "https://data.rcsb.org/graphql"
    # Define the query as a multi-line string with a variable for pdb_id
    entity_id_query: str = '["' + '", "'.join(entities) + '"]'

    # Prepare the request with the pdb_id as a variable
    query = QUERY_TEMPLATE.replace("$entity_ids", entity_id_query)

    # Make the request to the GraphQL endpoint using the variables
    response = requests.post(url, json={"query": query})

    # Check if the request was successful
    out: dict[str, float] = {}
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        # Loop through each nonpolymer entity and its instances
        if data["data"]:
            for entity in data["data"]["nonpolymer_entities"]:
                rcsb_id = entity["rcsb_id"]
                instances = entity["nonpolymer_entity_instances"]
                for instance in instances:
                    ranks = [
                        score["ranking_model_fit"]
                        for score in instance["rcsb_nonpolymer_instance_validation_score"]
                    ]
                    ranks = [r for r in ranks if r is not None]
                    if len(ranks) > 0:
                        out[rcsb_id] = max(ranks)
        else:
            warnings.warn(f"No data found for query: {query}")
    else:
        warnings.warn(
            f"Query failed to run by returning code of {response.status_code}.\n"
            f"Query: "
            f"{query}",
        )
    # remove entries with None values
    out = {k: v for k, v in out.items() if v is not None}
    return out
