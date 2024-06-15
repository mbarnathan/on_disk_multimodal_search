# Multimodal Semantic Search

semantic_search.ipynb is an on-disk multimodal semantic search pipeline.
It can be run locally with 4 images/second indexing throughput and 50-100ms query latency on a single RTX 3080 GPU. The pipeline also implements batching for greater throughput in a datacenter.

## Requirements
You will need Python 3.10+ (`segmentation_mask_overlay` makes use of a 3.10 language feature).
The "images" and "masks" subdirectories must exist and "images" must contain a set of image files.

Prior to running the notebook, ensure that all requirements are installed:
```pip install -r requirements.txt```

You will also want to ensure version 11.8 of the CUDA Toolkit is installed for GPU acceleration.

## How to Run

Two notebook cells are provided. Simply place a dataset of images in the images subdirectory and run the first cell to index.
Indexing progress will be displayed. The indexing may be interrupted at any time. Re-running will update existing image embeddings and insert new ones.

The indexing results are stored in a chromadb instance called "tags.db" in the local directory.

The second cell can be run to query the database. Once indexing has been performed,
simply enter a query phrase in natural language and run the notebook.

The top 10 results (by default) will be displayed, as well as any segmented objects of interest within the images.

## Configuration

All user-configurable variables are at the top of their respective cells.

Indexing utilizes these configuration parameters:
- `BATCH_SIZE`: Size of the image batch to process. (default: 2)
- `MAX_FILES`: Maximum number of images to process. (default: 1000)
- `MIN_SEG_SIZE`: Minimum pixel size a region must occupy to obtain a label. (default: 30)

Querying utilizes these configuration parameters:
- `TEXT_QUERY`: A natural language query you'd like to search against (e.g. "a cozy cottage with lots of toys")
- `MAX_RESULTS`: Number of results to return. (default: 10)

## Enhancements

Since this exercise was timeboxed, there are several opportunities for enhancement:

- **Image Segmentation**: I have benchmarked the Segment Anything Model coupled with an automatic seeding technique, and it is likely to result in a step change in performance.
- **Multimodal Loss**: Most of the complexity in this pipeline will disappear if a multimodal loss such as CLIP/CLIPSeg is used to index. Sentence embeddings will no longer need to be stored because they can be referenced against CLIP image embeddings directly. However, I have used this approach in the past and found it to be inferior to models such as BEiT.
- **Streaming**: pre-seeding SAM with something truly fast such as YOLOv8 would result in throughput sufficient to capture video in realtime. However, off-the-shelf YOLO models are primarily trained on COCO and do not generalize to many more classes.
- **Serving**: I'd write a Flask frontend if I wanted to expose an API. A Triton inference server would allow for greater parallelism as well.
- **Actions**: Segmentation alone isn't enough to characterize video. Context is required. An approach like [this one](https://news.mit.edu/2024/ai-based-method-can-find-specific-video-action-0529) looks promising.
