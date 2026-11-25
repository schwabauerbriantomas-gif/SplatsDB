//! Preset configurations for SplatsDB.
//!
//! Convenience constructors for common deployment scenarios.

use super::detect_device;
use super::types::*;
use super::SplatsDBConfig;

impl SplatsDBConfig {
    /// Handle device auto-detection and flags.
    pub fn finalize(&mut self) {
        if self.device == "auto" {
            self.device = detect_device().to_string();
        }

        match self.device.as_str() {
            "vulkan" => self.enable_vulkan = true,
            "cuda" => self.enable_cuda = true,
            _ => {}
        }
    }

    /// Creates a 'Simple' configuration for edge computing.
    ///
    /// Stripped-down: no quantization overhead, no graph, no auto-scaling.
    /// Pure HRM2 search with RAM-only memory.
    pub fn simple(device: Option<&str>) -> Self {
        let mut config = Self::default();

        if let Some(d) = device {
            config.device = d.to_string();
        }

        config.enable_3_tier_memory = false;
        config.memory_tier = "ram-only".to_string();
        config.splat_temperature = 0.0;

        // Disable heavy features
        config.enable_quantization = false;
        config.enable_graph = false;
        config.enable_mapreduce = false;
        config.enable_auto_scaling = false;
        config.enable_quality_reflection = false;
        config.enable_hnsw = false;
        config.enable_lsh = false;

        // Reduce HRM2 overhead
        config.hrm2_n_coarse = 8;
        config.hrm2_n_fine = 32;
        config.hrm2_n_probe = 2;
        config.max_splats = 10000;
        config.n_splats_init = 1000;
        config.knn_k = 32;

        // Reduce memory footprint
        config.langevin_steps = 0;
        config.context_local = 8;
        config.context_medium = 32;
        config.context_global = 128;
        config.hidden_dim = 512;
        config.moe_experts = 2;
        config.moe_active = 1;

        config
    }

    /// Creates an 'Advanced' configuration for Agentic AI workloads.
    ///
    /// Everything enabled: TurboQuant compression, GraphSplat knowledge graph,
    /// Semantic Memory with hybrid BM25+vector, auto-scaling, quality reflection.
    pub fn advanced(device: Option<&str>) -> Self {
        let mut config = Self::default();

        if let Some(d) = device {
            config.device = d.to_string();
        }

        config.enable_3_tier_memory = true;
        config.memory_tier = "3-tier".to_string();
        config.splat_temperature = 0.1;

        // Full quantization with TurboQuant for memory efficiency
        config.enable_quantization = true;
        config.quant_algorithm = QuantAlgorithm::TurboQuant;
        config.quant_bits = 4;
        config.quant_projections = 80;
        config.quant_search_fraction = 0.2;

        // GraphSplat for knowledge graph augmentation
        config.enable_graph = true;
        config.graph_max_neighbors = 20;
        config.graph_traverse_depth = 5;
        config.graph_boost_weight = 0.1;

        // Semantic Memory with RRF fusion and temporal decay
        config.semantic_fusion = FusionMethod::Rrf;
        config.semantic_vector_weight = 0.6;
        config.semantic_bm25_weight = 0.4;
        config.semantic_decay_halflife = 3600.0; // 1 hour

        // HNSW as secondary index for exact recall
        config.enable_hnsw = true;
        config.hnsw_m = 32;
        config.hnsw_ef_construction = 400;
        config.hnsw_ef_search = 100;

        // MapReduce for bulk indexing
        config.enable_mapreduce = true;
        config.mapreduce_n_chunks = 16;

        config.enable_auto_scaling = true;
        config.autoscale_min_nodes = 2;
        config.autoscale_max_nodes = 20;
        config.autoscale_cooldown_secs = 30.0;

        // Quality reflection with high recall target
        config.enable_quality_reflection = true;
        config.quality_recall_target = 0.99;

        // Larger capacity
        config.max_splats = 1_000_000;
        config.n_splats_init = 50000;
        config.knn_k = 128;
        config.hrm2_n_coarse = 32;
        config.hrm2_n_fine = 128;
        config.hrm2_n_probe = 8;

        // Full context windows
        config.context_local = 24;
        config.context_medium = 128;
        config.context_global = 1024;

        config
    }

    /// Creates an 'MCP' configuration optimized for AI agent memory workloads.
    ///
    /// GPU-accelerated when available, moderate capacity, all smart features on.
    /// Designed for the MCP server use case: store/search/recall with embeddings.
    #[allow(clippy::field_reassign_with_default)]
    pub fn mcp(device: Option<&str>) -> Self {
        let mut config = Self::default();

        config.device = device.unwrap_or("auto").to_string();
        config.finalize();

        // GPU acceleration when available
        config.enable_gpu_search = config.enable_cuda;
        config.gpu_batch_size = 1024;

        // Moderate capacity for agent memory
        config.max_splats = 100_000;
        config.n_splats_init = 10_000;
        config.knn_k = 64;

        // HRM2 for fast approximate search
        config.hrm2_n_coarse = 32;
        config.hrm2_n_fine = 128;
        config.hrm2_n_probe = 8;

        // Quantization for memory efficiency
        config.enable_quantization = true;
        config.quant_algorithm = QuantAlgorithm::TurboQuant;
        config.quant_bits = 8;

        // Knowledge graph for semantic connections
        config.enable_graph = true;
        config.graph_max_neighbors = 20;
        config.graph_traverse_depth = 5;

        // Semantic memory with hybrid search
        config.semantic_fusion = FusionMethod::Rrf;
        config.semantic_vector_weight = 0.6;
        config.semantic_bm25_weight = 0.4;

        config
    }

    /// Configuration for the local Hermes memory hub (`serve --profile hermes`).
    ///
    /// The hub's hot path is HNSW candidate generation with exact GPU L2
    /// re-rank over 768-d session/code embeddings. Keep that contract explicit:
    /// the server preset owns HNSW tuning, while graph, map-reduce, training,
    /// and quality-reflection machinery stay off because those paths are unused.
    /// Device detection is explicit (`auto` => CUDA when available, else CPU).
    pub fn hermes(device: Option<&str>) -> Self {
        let mut config = Self {
            device: device.unwrap_or("auto").to_string(),
            cluster_trainer: crate::clustering::TrainerKind::Emeans,
            enable_quantization: false,
            enable_graph: false,
            enable_hnsw: true,
            enable_lsh: false,
            enable_mapreduce: false,
            enable_auto_scaling: false,
            enable_quality_reflection: false,
            enable_training: false,
            enable_data_lake: false,
            enable_entity_extraction: false,
            ..Self::default()
        };
        config.finalize();
        config
    }

    /// Creates a 'GPU' configuration for CUDA-accelerated search.
    ///
    /// Enables GPU search backends, auto-tuning, and large batch operations.
    /// Falls back to CPU if no GPU is available.
    #[allow(clippy::field_reassign_with_default)]
    pub fn gpu(device: Option<&str>) -> Self {
        let mut config = Self::default();

        config.device = device.unwrap_or("cuda").to_string();
        config.enable_cuda = true;
        config.enable_vulkan = false;

        // GPU search acceleration
        config.enable_gpu_search = true;
        config.gpu_batch_size = 4096;

        // HNSW on GPU for high-throughput ANN
        config.enable_hnsw = true;
        config.hnsw_m = 48;
        config.hnsw_ef_construction = 800;
        config.hnsw_ef_search = 200;

        // Full quantization pipeline
        config.enable_quantization = true;
        config.quant_algorithm = QuantAlgorithm::TurboQuant;
        config.quant_bits = 4;
        config.quant_projections = 160;
        config.quant_search_fraction = 0.05;

        // Graph and semantic at scale
        config.enable_graph = true;
        config.graph_max_neighbors = 32;
        config.graph_traverse_depth = 5;

        // Large capacity with GPU memory
        config.max_splats = 5_000_000;
        config.n_splats_init = 100_000;
        config.knn_k = 256;
        config.hrm2_n_coarse = 64;
        config.hrm2_n_fine = 256;
        config.hrm2_n_probe = 16;

        config
    }
}

#[cfg(test)]
mod hermes_profile_tests {
    use super::*;

    #[test]
    fn hermes_profile_is_lean_and_serializable() {
        let config = SplatsDBConfig::hermes(Some("cpu"));
        assert!(!config.enable_cuda);
        assert!(!config.enable_quantization);
        assert!(!config.enable_graph);
        assert!(config.enable_hnsw);
        assert!(!config.enable_lsh);
        assert!(!config.enable_mapreduce);
        assert!(!config.enable_quality_reflection);
        assert_eq!(
            config.cluster_trainer,
            crate::clustering::TrainerKind::Emeans
        );
        let json = serde_json::to_string(&config).expect("serialize hermes profile");
        let restored: SplatsDBConfig =
            serde_json::from_str(&json).expect("deserialize hermes profile");
        assert_eq!(restored.cluster_trainer, config.cluster_trainer);
        assert!(!restored.enable_quantization);
    }

    #[test]
    fn hermes_profile_honors_cuda_device() {
        let config = SplatsDBConfig::hermes(Some("cuda"));
        assert!(config.enable_cuda);
        assert!(!config.enable_vulkan);
    }

    #[test]
    fn default_profile_remains_unchanged() {
        let config = SplatsDBConfig::default();
        assert!(config.enable_quantization);
        assert!(config.enable_graph);
        assert!(!config.enable_hnsw);
        assert_eq!(
            config.cluster_trainer,
            crate::clustering::TrainerKind::Lloyd
        );
    }
}
