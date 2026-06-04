document.addEventListener("DOMContentLoaded", () => {
    // 1. Core Documentation Search Index
    const searchIndex = [
        { title: "00. Overview: The Journey of a Token", url: "index.html", desc: "How a piece of text becomes an LLM response. Mentally maps out input to token predictions." },
        { title: "01. Tokenization: Payload Parsing", url: "tokenization.html", desc: "Byte Pair Encoding (BPE), subword tokenization, tiktoken usage, and translating strings to primary keys (IDs)." },
        { title: "02. Embeddings: Mapping to Vector Space", url: "architecture.html", desc: "Token and position embeddings (wte & wpe), dynamic lookup directories, dimensions, and broadcasting." },
        { title: "03. Attention: The Context Index", url: "attention.html", desc: "Causal Multi-Head Self-Attention, Query-Key-Value similarity search analogy, causal masking, and FlashAttention." },
        { title: "04. MLP & LayerNorm: Processing Pipelines", url: "transformer_block.html", desc: "Transformer blocks, feed-forward networks (MLP), standardizing inputs (LayerNorm), and residual context retention middleware." },
        { title: "05. The Output: Next-Token Prediction", url: "generation.html", desc: "LM head projections, next-token probabilities, temperature scaling, top-k sampling, and autoregressive generation loops." },
        { title: "06. Data Loader: Streaming Batches", url: "data_pipeline.html", desc: "DataLoaderLite batches, context window slices, shifted targets as state logs, and epoch wraps." },
        { title: "07. Backpropagation: The Feedback Loop", url: "training_loop.html", desc: "AdamW parameter weight decay groups, gradient accumulation batch processing, and cosine warmup schedules." },
        { title: "08. Hardware: Mixed Precision & Speedups", url: "optimization.html", desc: "Apple Silicon MPS synchronizations, mixed-precision FP16 (AMP) datatype compression, and torch.compile graph compilation." },
        { title: "09. Checkpoints: Application Snapshots", url: "checkpointing.html", desc: "Serialization mappings, config and optimizer state dicts, and Conv1D weight transpositions." },
        { title: "10. Evaluation: Performance Validation", url: "evaluation.html", desc: "Validation loss tracking, HellaSwag benchmark completions, and option log-likelihood scoring." },
        { title: "11. The Tensor Shape Journey", url: "tensor_journey.html", desc: "Trace batch shape dimensions [B, T, C] through the embedding, attention, MLP, and output projection layers." },
        { title: "12. Memory Audit: Where Memory Goes", url: "memory_performance.html", desc: "Train memory footprint audit: dividing memory between parameters, gradients, optimizer states, and forward activations." },
        { title: "13. Failure Gallery: Debugging & Troubleshooting", url: "debugging.html", desc: "Common deep learning bugs including NaNs, exploding gradients, mask mismatches, and shape crashes." },
        { title: "14. Experiment Lab: Structural Modifications", url: "experiments.html", desc: "Ablation tests: removing residual streams, removing position embeddings, width adjustments, and custom activations." },
        { title: "15. GPT-2 to Modern LLMs: Architecture Evolution", url: "modern_llms.html", desc: "Evolution comparison of GPT-2 to Llama, Gemma, and Mistral covering RoPE, RMSNorm, GQA, SwiGLU, and serving architectures (vLLM, LoRA)." },
        { title: "16. Interview Guide: Grilling the LLM Engineer", url: "interview_intent.html", desc: "System design and implementation questions commonly asked in GenAI systems and LLM engineering interviews." },
        { title: "17. How to Read the Codebase", url: "code_guide.html", desc: "Step-by-step reading order guide for model.py, generate.py, train.py, and hellaswag.py." },
        { title: "18. Learning Roadmap: Next Steps", url: "engineer_roadmap.html", desc: "GenAI developer roadmap scaling from toy GPTs to fine-tuning, RAG, LLMOps, and serving systems." },
        { title: "19. Glossary of Terms", url: "glossary.html", desc: "Key transformer terminology including KV cache, Perplexity, FLOPs, weight tying, and mixed precision." }
    ];

    // 2. Setup Search Overlay Elements
    const searchModal = document.createElement("div");
    searchModal.className = "search-modal";
    searchModal.id = "search-modal";
    searchModal.innerHTML = `
        <div class="search-modal-content">
            <div class="search-modal-header">
                <input type="text" class="search-modal-input" id="search-modal-input" placeholder="Search the handbook..." autocomplete="off">
                <button class="search-modal-close" id="search-modal-close">&times;</button>
            </div>
            <ul class="search-results-list" id="search-results-list"></ul>
        </div>
    `;
    document.body.appendChild(searchModal);

    const searchInput = document.getElementById("search-input");
    const modalInput = document.getElementById("search-modal-input");
    const closeBtn = document.getElementById("search-modal-close");
    const resultsList = document.getElementById("search-results-list");

    const openSearch = () => {
        searchModal.style.display = "flex";
        modalInput.value = "";
        resultsList.innerHTML = "";
        setTimeout(() => modalInput.focus(), 50);
    };

    const closeSearch = () => {
        searchModal.style.display = "none";
    };

    // Keyboard bindings (slash '/' or cmd/ctrl + K to open search)
    document.addEventListener("keydown", (e) => {
        if (e.key === "/" && document.activeElement !== searchInput && document.activeElement !== modalInput) {
            e.preventDefault();
            openSearch();
        }
        if ((e.metaKey || e.ctrlKey) && e.key === "k") {
            e.preventDefault();
            openSearch();
        }
        if (e.key === "Escape") {
            closeSearch();
        }
    });

    if (searchInput) {
        searchInput.addEventListener("click", openSearch);
    }
    if (closeBtn) {
        closeBtn.addEventListener("click", closeSearch);
    }

    // Modal background click closes search
    searchModal.addEventListener("click", (e) => {
        if (e.target === searchModal) {
            closeSearch();
        }
    });

    // 3. Search Processing
    const runSearch = (query) => {
        resultsList.innerHTML = "";
        if (!query.trim()) return;

        const terms = query.toLowerCase().split(/\s+/);
        const matches = searchIndex.filter(item => {
            const text = (item.title + " " + item.desc).toLowerCase();
            return terms.every(term => text.includes(term));
        });

        if (matches.length === 0) {
            resultsList.innerHTML = `<li style="padding: 16px; text-align: center; color: var(--text-secondary); font-size: 0.9rem;">No results found for "${query}"</li>`;
            return;
        }

        matches.forEach(item => {
            const li = document.createElement("li");
            li.className = "search-result-item";
            li.innerHTML = `
                <a href="${item.url}" class="search-result-link">
                    <div class="search-result-title">${item.title}</div>
                    <div class="search-result-snippet">${item.desc}</div>
                </a>
            `;
            resultsList.appendChild(li);
        });
    };

    modalInput.addEventListener("input", (e) => {
        runSearch(e.target.value);
    });
});
