document.addEventListener("DOMContentLoaded", () => {
    // 1. Core Documentation Search Index
    const searchIndex = [
        { title: "Home Dashboard", url: "index.html", desc: "Overview, hyperparameters, codebase structure, and local hardware requirements." },
        { title: "Architecture: Token-to-Loss Lifecycle", url: "architecture.html", desc: "Step-by-step activations, token embeddings, layer normalizations, tensor shape dimensions, parameters, and FLOP math." },
        { title: "Causal Multi-Head Self-Attention", url: "attention.html", desc: "QKV mathematical derivation, attention scores, causal masking, MHA heads, and FlashAttention memory bandwidth optimization." },
        { title: "Transformer Block Internals", url: "transformer_block.html", desc: "Pre-LN vs Post-LN structures, feed-forward expansion layers, GELU activation, residual scaling, and initialization logic." },
        { title: "Tokenization & Byte Pair Encoding", url: "tokenization.html", desc: "BPE theory, tiktoken usage, vocabulary limits, out-of-vocabulary fallback, and parameter-tying lm_head." },
        { title: "DataLoaderLite & Splits", url: "data_pipeline.html", desc: "Offset batching calculations, wrap-around index boundary checks, and train/val Shakespeare splits." },
        { title: "Training Loop: Mechanics & Schedule", url: "training_loop.html", desc: "Gradient accumulation scaling, weight decay optimization, AdamW decay groups, and cosine learning rate warmup schedules." },
        { title: "Optimization: Apple Silicon & Compilers", url: "optimization.html", desc: "MPS device synchronizations, AMP autocasting mixed-precision training, torch.compile graph optimization, and bottlenecks." },
        { title: "Autoregressive Generation", url: "generation.html", desc: "Generative loop, cropping context lengths, temperature scaling calculations, top-k filtering, and multinomial samplers." },
        { title: "Checkpointing & Restoration", url: "checkpointing.html", desc: "Serialization mappings, config and optimizer tracking, and CPU-first deterministic weight restoration patterns." },
        { title: "Evaluation & validation", url: "evaluation.html", desc: "Validation losses, HellaSwag evaluation, log-likelihood calculations, and context token ignore mappings." },
        { title: "Structural Experiments", url: "experiments.html", desc: "Systematic experiments (removing residuals, position embeddings, modifying width, custom activations)." },
        { title: "Connections to Modern LLMs", url: "modern_llms.html", desc: "Evolution comparison of GPT-2 to Llama, Gemma, DeepSeek, Mistral, Claude, and GPT-4 covering RoPE, RMSNorm, GQA, SwiGLU, and MoE." },
        { title: "Interview Preparation Guide", url: "interview_guide.html", desc: "Beginner-to-staff level transformer QA database, pitfalls, and technical follow-ups." },
        { title: "Glossary of Terms", url: "glossary.html", desc: "Core terminology definitions including KV cache, Perplexity, FLOPs, weight tying, and BF16." }
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
