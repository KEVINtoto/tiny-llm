//! Maintainer guards for the Rust starter and the published course files.
//!
//! Rust declarations are checked with syn, rather than identifier substrings.
//! The supplied CLI and capstone are still Python files; their static guards
//! inspect those actual artifacts. No Python reference implementation is run.
#[path = "support/temp.rs"]
mod test_temp;

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use syn::parse::Parser;
use syn::{ImplItem, Item, Visibility};
use test_temp::tempdir;
use tiny_llm_agent::generation::Message;
use tiny_llm_agent::workspace::ConfirmResult;
use tiny_llm_agent::{
    ArtifactStore, BoundedEvidenceWorkspace, ReceiptStore, ToolAction, ToolPolicy, Workspace,
};
use tiny_llm_agent::{checkpoint, generation, protocol};

// Compile the original helper as test-local source so its private observation
// boundary can be mutation-tested without changing agent/lib.rs or loop.rs.
#[allow(dead_code, unused_variables)]
#[path = "../loop.rs"]
mod learner_loop;

const MODULES: [&str; 11] = [
    "branching",
    "checkpoint",
    "compaction",
    "evidence",
    "evaluation",
    "generation",
    "loop",
    "protocol",
    "receipts",
    "steering",
    "workspace",
];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}
fn source(module: &str) -> String {
    fs::read_to_string(root().join("agent").join(format!("{module}.rs"))).unwrap()
}
fn tree(module: &str) -> syn::File {
    syn::parse_file(&source(module)).unwrap()
}
fn names(values: &[&str]) -> BTreeSet<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}
fn public_names(module: &str) -> BTreeSet<String> {
    tree(module)
        .items
        .into_iter()
        .filter_map(|item| {
            let (vis, name) = match item {
                Item::Struct(i) => (i.vis, i.ident),
                Item::Enum(i) => (i.vis, i.ident),
                Item::Trait(i) => (i.vis, i.ident),
                Item::Type(i) => (i.vis, i.ident),
                Item::Static(i) => (i.vis, i.ident),
                Item::Fn(i) => (i.vis, i.sig.ident),
                _ => return None,
            };
            matches!(vis, Visibility::Public(_)).then(|| name.to_string())
        })
        .collect()
}
fn methods(module: &str, class: &str) -> BTreeSet<String> {
    let mut result = BTreeSet::new();
    for item in tree(module).items {
        if let Item::Impl(item) = item {
            if item.trait_.is_some() {
                continue;
            }
            let syn::Type::Path(path) = *item.self_ty else {
                continue;
            };
            if !path.path.is_ident(class) {
                continue;
            }
            for member in item.items {
                if let ImplItem::Fn(method) = member
                    && matches!(method.vis, Visibility::Public(_))
                {
                    result.insert(method.sig.ident.to_string());
                }
            }
        }
    }
    result
}
fn assert_methods(module: &str, class: &str, expected: &[&str]) {
    assert_eq!(methods(module, class), names(expected), "{module}::{class}");
}
fn public_fields(module: &str, class: &str) -> BTreeSet<String> {
    for item in tree(module).items {
        if let Item::Struct(item) = item
            && item.ident == class
        {
            return item
                .fields
                .into_iter()
                .filter(|f| matches!(f.vis, Visibility::Public(_)))
                .map(|f| f.ident.expect("named public field").to_string())
                .collect();
        }
    }
    panic!("missing {module}::{class}");
}
fn assert_no_future(names: &[&str]) {
    let joined = MODULES.map(source).join("\n");
    let identifiers: BTreeSet<_> = joined
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .collect();
    for name in names {
        assert!(!identifiers.contains(name), "unexpected future API {name}");
    }
}
fn root_exports() -> BTreeSet<String> {
    fn collect(tree: syn::UseTree, names: &mut BTreeSet<String>) {
        match tree {
            syn::UseTree::Path(path) => collect(*path.tree, names),
            syn::UseTree::Name(name) => {
                names.insert(name.ident.to_string());
            }
            syn::UseTree::Rename(rename) => {
                names.insert(rename.rename.to_string());
            }
            syn::UseTree::Group(group) => {
                for item in group.items {
                    collect(item, names);
                }
            }
            syn::UseTree::Glob(_) => panic!("root exports must remain explicit"),
        }
    }
    let mut result = BTreeSet::new();
    for item in tree("lib").items {
        if let Item::Use(item) = item
            && matches!(item.vis, Visibility::Public(_))
        {
            collect(item.tree, &mut result);
        }
    }
    result
}

#[test]
fn test_starter_public_contract_matches_reference() {
    // Names are the published Python surface plus explicit Rust support types.
    let expected: [(&str, &[&str]); 11] = [
        (
            "branching",
            &[
                "PrefixReuse",
                "BranchOutcome",
                "KvPrefixGenerator",
                "run_branch",
                "select_branch",
            ],
        ),
        (
            "checkpoint",
            &["ModelCheckpoint", "AgentCheckpoint", "create_checkpoint"],
        ),
        (
            "compaction",
            &["CompactionResult", "compact_completed_interactions"],
        ),
        (
            "evidence",
            &["ArtifactRef", "ArtifactStore", "BoundedEvidenceWorkspace"],
        ),
        (
            "evaluation",
            &[
                "FileExpectation",
                "ResultExpectation",
                "ReceiptExpectation",
                "ReceiptLookup",
                "EvaluationCase",
                "EvaluationCheck",
                "EvaluationReport",
                "evaluate_run",
            ],
        ),
        (
            "generation",
            &[
                "Message",
                "Generate",
                "GenerationCache",
                "GenerationModel",
                "GenerationTokenizer",
                "initial_messages",
                "generate_response",
            ],
        ),
        (
            "loop",
            &[
                "AgentLimits",
                "AgentEvent",
                "AgentRun",
                "run_agent",
                "run_to_checkpoint",
                "resume_agent",
            ],
        ),
        (
            "protocol",
            &[
                "AgentError",
                "FinalAction",
                "ToolAction",
                "AgentAction",
                "AgentWorkspace",
                "TOOL_FIELDS",
                "parse_action",
                "build_system_prompt",
            ],
        ),
        ("receipts", &["EffectReceipt", "ReceiptStore"]),
        (
            "steering",
            &["AgentStatus", "inspect_checkpoint", "resume_with_steering"],
        ),
        (
            "workspace",
            &[
                "ApprovalDecision",
                "ConfirmResult",
                "ConfirmTool",
                "ToolPolicy",
                "Workspace",
            ],
        ),
    ];
    for (module, expected) in expected {
        assert_eq!(public_names(module), names(expected), "{module}");
        assert!(
            root()
                .join(format!("src/tiny_llm_ref/agent/{module}.py"))
                .is_file()
        );
    }
    // Pin the callable model boundary shared by Day 1 and Day 8.
    let _: fn(&str, &str) -> Result<Vec<Message>, tiny_llm_agent::AgentError> =
        tiny_llm_agent::initial_messages;
    type GenerateResponse = fn(
        &mut dyn generation::GenerationModel,
        &mut dyn generation::GenerationTokenizer,
        &[Message],
        &mut dyn FnMut() -> Vec<Box<dyn generation::GenerationCache>>,
        i64,
        bool,
    ) -> Result<String, tiny_llm_agent::AgentError>;
    let _: GenerateResponse = tiny_llm_agent::generate_response;
    let _: for<'a> fn(&'a ReceiptStore, &str) -> Option<&'a tiny_llm_agent::EffectReceipt> =
        ReceiptStore::get;
}

#[test]
fn test_starter_is_solution_free_and_reference_is_implemented() {
    // A starter guard, not a requirement to fill in the learner's solutions.
    fn stub(block: &syn::Block) -> bool {
        block.stmts.iter().any(|stmt| match stmt {
            syn::Stmt::Macro(m) => m.mac.path.is_ident("todo"),
            syn::Stmt::Expr(syn::Expr::Macro(m), _) => m.mac.path.is_ident("todo"),
            _ => false,
        })
    }
    for module in MODULES {
        for item in tree(module).items {
            match item {
                Item::Fn(function) if matches!(function.vis, Visibility::Public(_)) => {
                    assert!(
                        stub(&function.block),
                        "{module}::{} contains solution logic",
                        function.sig.ident
                    );
                }
                Item::Impl(item) if item.trait_.is_none() => {
                    let syn::Type::Path(path) = *item.self_ty else {
                        continue;
                    };
                    let class = path.path.segments.last().unwrap().ident.to_string();
                    for member in item.items {
                        if let ImplItem::Fn(method) = member {
                            if !matches!(method.vis, Visibility::Public(_)) {
                                continue;
                            }
                            let name = method.sig.ident.to_string();
                            // Supplied constructors and Day 9's range-cap validation
                            // are scaffold plumbing already present before migration.
                            if name == "new"
                                || (class == "BoundedEvidenceWorkspace" && name == "post_init")
                            {
                                continue;
                            }
                            assert!(
                                stub(&method.block),
                                "{module}::{class}::{name} contains solution logic"
                            );
                        }
                    }
                }
                _ => {}
            }
        }
        let reference =
            fs::read_to_string(root().join(format!("src/tiny_llm_ref/agent/{module}.py"))).unwrap();
        assert!(
            reference.contains("def "),
            "{module} reference has no functions"
        );
        assert!(
            reference.contains("return ") || reference.contains("raise "),
            "{module} reference is not implemented"
        );
    }
}

#[test]
fn test_package_exports_match_the_published_day_9_surface() {
    assert_eq!(
        root_exports(),
        names(&[
            "AgentError",
            "AgentCheckpoint",
            "AgentEvent",
            "AgentLimits",
            "AgentRun",
            "AgentStatus",
            "ApprovalDecision",
            "ArtifactRef",
            "ArtifactStore",
            "BoundedEvidenceWorkspace",
            "BranchOutcome",
            "CompactionResult",
            "EvaluationCase",
            "EvaluationCheck",
            "EvaluationReport",
            "EffectReceipt",
            "FinalAction",
            "FileExpectation",
            "KvPrefixGenerator",
            "ModelCheckpoint",
            "ReceiptStore",
            "ReceiptExpectation",
            "ResultExpectation",
            "PrefixReuse",
            "ToolAction",
            "ToolPolicy",
            "Workspace",
            "build_system_prompt",
            "compact_completed_interactions",
            "create_checkpoint",
            "evaluate_run",
            "generate_response",
            "initial_messages",
            "inspect_checkpoint",
            "parse_action",
            "resume_agent",
            "resume_with_steering",
            "run_agent",
            "run_branch",
            "run_to_checkpoint",
            "select_branch",
        ])
    );
}

#[test]
fn test_only_day_1_through_day_9_modules_exist_in_the_starter() {
    let actual: BTreeSet<_> = fs::read_dir(root().join("agent"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "rs"))
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    let expected: BTreeSet<_> = MODULES
        .iter()
        .copied()
        .chain(["lib"])
        .map(|m| format!("{m}.rs"))
        .collect();
    assert_eq!(actual, expected);
}

#[test]
fn test_day_3_workspace_surface_remains_complete() {
    assert_methods(
        "workspace",
        "Workspace",
        &[
            "new",
            "available_tools",
            "modified_files",
            "resolve_path",
            "list_files",
            "read_file",
            "write_file",
            "edit_file",
            "run_command",
            "execute",
        ],
    );
    assert_methods("workspace", "ToolPolicy", &["new", "post_init"]);
    assert_methods("workspace", "ApprovalDecision", &["new", "post_init"]);
    assert_methods(
        "receipts",
        "ReceiptStore",
        &["new", "post_init", "put", "get"],
    );
    assert_methods(
        "receipts",
        "EffectReceipt",
        &["new", "post_init", "receipt_id", "to_dict", "from_dict"],
    );
}

#[test]
fn test_day_4_checkpoint_surface_remains_complete() {
    assert_methods("checkpoint", "AgentCheckpoint", &["validate"]);
    assert_methods("checkpoint", "ModelCheckpoint", &["new", "post_init"]);
    assert_eq!(
        public_fields("checkpoint", "AgentCheckpoint"),
        names(&["checkpoint_id", "task", "messages", "model"])
    );
    assert!(public_names("loop").is_superset(&names(&["run_to_checkpoint", "resume_agent"])));
    assert_no_future(&["Session", "rewind"]);
}

#[test]
fn test_day_5_compaction_surface_is_complete() {
    assert_methods("compaction", "CompactionResult", &["saved_tokens"]);
    assert_eq!(
        public_fields("compaction", "CompactionResult"),
        names(&[
            "messages",
            "compacted_interactions",
            "tokens_before",
            "tokens_after",
            "receipt_ids"
        ])
    );
}

#[test]
fn test_day_6_steering_surface_remains_complete() {
    assert_eq!(
        public_names("steering"),
        names(&["AgentStatus", "inspect_checkpoint", "resume_with_steering"])
    );
    assert_eq!(
        public_fields("steering", "AgentStatus"),
        names(&["task", "last_action", "last_evidence", "next_step"])
    );
}

#[test]
fn test_day_7_evaluation_surface_is_complete_and_has_no_future_api() {
    for class in [
        "EvaluationCase",
        "FileExpectation",
        "ReceiptExpectation",
        "ResultExpectation",
    ] {
        assert_methods("evaluation", class, &["new", "post_init"]);
    }
    assert_methods("evaluation", "EvaluationReport", &["passed", "render"]);
    assert_no_future(&["Session", "rewind", "reconcile", "llm_judge", "radix"]);
}

#[test]
fn test_day_8_branching_surface_is_complete_and_has_no_future_api() {
    assert_methods(
        "branching",
        "KvPrefixGenerator",
        &[
            "new",
            "fork",
            "restore_checkpoint",
            "reuse",
            "save_checkpoint",
            "call",
        ],
    );
    assert_eq!(
        public_fields("branching", "BranchOutcome"),
        names(&["name", "steering", "run", "report", "reuse"])
    );
    assert_eq!(
        public_fields("branching", "PrefixReuse"),
        names(&["reused_tokens", "layer_offsets", "avoided_prefill_tokens"])
    );
    assert_no_future(&["Session", "reconcile", "radix"]);
}

#[test]
fn test_day_9_bounded_evidence_surface_is_complete_and_has_no_future_api() {
    assert_methods("evidence", "ArtifactRef", &["new", "post_init"]);
    assert_methods(
        "evidence",
        "ArtifactStore",
        &["new", "post_init", "put", "range_path", "read_range"],
    );
    assert_methods(
        "evidence",
        "BoundedEvidenceWorkspace",
        &[
            "new",
            "post_init",
            "available_tools",
            "execute",
            "modified_files",
            "policy",
        ],
    );
    assert_no_future(&["SemanticSummary", "BlobService", "StreamingDispatcher"]);
}

#[test]
fn test_day_9_starter_range_cap_can_hold_one_max_width_utf8_character() {
    let temp = tempdir().unwrap();
    for max_range_bytes in [1, 2, 3, 4] {
        let policy = ToolPolicy {
            root: temp.path().to_path_buf(),
            max_file_bytes: 65536,
            max_list_entries: 200,
            allow_writes: false,
            allowed_commands: vec![],
            max_write_bytes: 65536,
            command_timeout_seconds: 30.0,
        };
        let workspace = Workspace::new(policy, None, ReceiptStore::new(None).unwrap());
        let artifacts = ArtifactStore::new(temp.path().to_path_buf()).unwrap();
        let result = BoundedEvidenceWorkspace::new(workspace, artifacts, 512, 64, max_range_bytes);
        if max_range_bytes < 4 {
            let error = result.err().expect("range cap below four must fail");
            assert!(
                error
                    .to_string()
                    .contains("max_range_bytes must be at least 4")
            );
        } else {
            assert_eq!(result.unwrap().max_range_bytes, 4);
        }
    }
}

#[test]
fn test_removed_catalog_hash_is_not_exported_or_declared() {
    for name in ["TOOL_CATALOG_HASH", "tool_catalog_hash"] {
        assert!(!source("protocol").contains(name));
        assert!(!root_exports().contains(name));
        let reference =
            fs::read_to_string(root().join("src/tiny_llm_ref/agent/protocol.py")).unwrap();
        assert!(!reference.contains(name));
    }
}

#[test]
fn test_starter_does_not_import_the_reference_solution() {
    for module in MODULES.into_iter().chain(["lib"]) {
        assert!(!source(module).contains("tiny_llm_ref"), "{module}");
    }
}

#[test]
fn test_real_model_cli_exposes_only_the_learner_package() {
    let text = fs::read_to_string(root().join("agent.py")).unwrap();
    assert!(
        text.lines()
            .any(|line| line == "from tiny_llm import agent")
    );
    for forbidden in ["tiny_llm_ref", "--solution", "importlib"] {
        assert!(!text.contains(forbidden));
    }
    assert!(text.contains("def build_parser()"));
    assert!(text.contains("parser.parse_args()"));
    assert!(text.contains("args.root"));
}

#[test]
fn test_real_model_cli_discloses_command_side_effect_scope() {
    let temp = tempdir().unwrap();
    let command = vec!["/usr/bin/touch".to_owned(), "extra".to_owned()];
    let mut workspace = Workspace::new(
        ToolPolicy::new(
            temp.path().to_path_buf(),
            65536,
            200,
            false,
            vec![command.clone()],
            65536,
            30.0,
        )
        .unwrap(),
        Some(Box::new(|_| ConfirmResult::Approved(true))),
        ReceiptStore::new(None).unwrap(),
    );
    let action = ToolAction {
        tool: "run_command".into(),
        arguments: serde_json::json!({"argv": command})
            .as_object()
            .unwrap()
            .clone(),
    };
    let result = workspace.execute(&action, None);
    assert!(temp.path().join("extra").is_file());
    assert!(result.starts_with("status: 0"));
    assert!(workspace.modified_files().is_empty());
    assert!(
        workspace
            .receipt_store
            .get("call-1")
            .unwrap()
            .changed_artifacts
            .is_empty()
    );
    // Inspect the published CLI's user-facing print literals, not Rust comments.
    let cli = fs::read_to_string(root().join("agent.py")).unwrap();
    assert!(cli.contains("print(\"file-tool changes> \""));
    assert!(cli.contains("approved commands may create or change additional filesystem "));
    assert!(cli.contains("paths; command receipts record the command and result, not a complete "));
    assert!(cli.contains("filesystem diff"));
}

#[test]
fn test_reference_observation_payload_mutations_are_killed() {
    let initial = vec![
        Message::from([
            ("role".into(), "system".into()),
            ("content".into(), "system".into()),
        ]),
        Message::from([
            ("role".into(), "user".into()),
            ("content".into(), "inspect".into()),
        ]),
    ];
    let observed = learner_loop::append_tool_result(
        &initial,
        r#"{"tool":"read_file","path":"README.md"}"#,
        "exact observable bytes\n",
    );
    let check = |messages: &[Message]| {
        assert_eq!(messages.last().unwrap()["role"], "user");
        assert_eq!(
            messages.last().unwrap()["content"],
            "Tool result:\nexact observable bytes\n"
        );
    };
    check(&observed);
    for mutation in ["Tool result:\nwrong payload", "Tool result:"] {
        let mut broken = observed.clone();
        broken
            .last_mut()
            .unwrap()
            .insert("content".into(), mutation.into());
        assert!(
            catch_unwind(AssertUnwindSafe(|| check(&broken))).is_err(),
            "exact observation guard must kill mutation {mutation:?}"
        );
    }
}

#[test]
fn test_capstone_is_learner_only_and_stale_evaluator_is_fully_removed() {
    let capstone = fs::read_to_string(root().join("week4-capstone.py")).unwrap();
    for forbidden in ["tiny_llm_ref", "--solution", "importlib"] {
        assert!(!capstone.contains(forbidden));
    }
    let public_functions: BTreeSet<_> = capstone
        .lines()
        .filter_map(|line| line.strip_prefix("def "))
        .filter_map(|line| line.split('(').next())
        .filter(|name| !name.starts_with('_'))
        .map(str::to_owned)
        .collect();
    assert_eq!(public_functions, names(&["run_capstone", "main"]));
    let project = fs::read_to_string(root().join("pyproject.toml")).unwrap();
    assert!(project.contains("week4-capstone"));
    assert!(!project.contains("evaluate-agent"));
    assert!(!root().join("evaluate-agent.py").exists());
    assert!(!root().join("evals/week4").exists());
}

#[test]
fn test_port_inventory_covers_all_python_tests() {
    let inventory: serde_json::Value = serde_json::from_str(include_str!("coverage.json")).unwrap();
    let mut recorded_sources = BTreeSet::new();
    for entry in inventory.as_array().unwrap() {
        let source = entry["source"].as_str().unwrap();
        recorded_sources.insert(source.to_owned());
        let python = fs::read_to_string(root().join(source)).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(python.as_bytes())),
            entry["source_sha256"].as_str().unwrap(),
            "Reference assertions changed; review Rust parity before updating the digest: {source}"
        );
        let declared: BTreeSet<_> = python
            .lines()
            .filter_map(|line| line.strip_prefix("def test_"))
            .map(|tail| format!("test_{}", tail.split('(').next().unwrap()))
            .collect();
        let recorded: BTreeSet<_> = entry["tests"]
            .as_array()
            .unwrap()
            .iter()
            .map(|test| test["python"].as_str().unwrap().to_owned())
            .collect();
        assert_eq!(
            declared, recorded,
            "Python test inventory drifted: {source}"
        );
        let rust = fs::read_to_string(root().join(entry["target"].as_str().unwrap())).unwrap();
        let mut runnable = BTreeMap::new();
        for item in syn::parse_file(&rust).unwrap().items {
            match item {
                Item::Fn(function) if function.attrs.iter().any(|a| a.path().is_ident("test")) => {
                    let name = function.sig.ident.to_string();
                    assert!(!function.block.stmts.is_empty(), "empty test: {name}");
                    if function.block.stmts.len() == 1 {
                        let placeholder = match &function.block.stmts[0] {
                            syn::Stmt::Macro(m) => Some(&m.mac.path),
                            syn::Stmt::Expr(syn::Expr::Macro(m), _) => Some(&m.mac.path),
                            _ => None,
                        };
                        assert!(
                            !placeholder.is_some_and(|p| p.is_ident("panic")
                                || p.is_ident("todo")
                                || p.is_ident("unimplemented")),
                            "placeholder test: {name}"
                        );
                    }
                    let ignored = function.attrs.iter().any(|a| a.path().is_ident("ignore"));
                    assert!(runnable.insert(name, ignored).is_none());
                }
                Item::Macro(item) if item.mac.path.is_ident("invalid_status_test") => {
                    let args =
                        syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated
                            .parse2(item.mac.tokens)
                            .unwrap();
                    let syn::Expr::Path(name) = &args[0] else {
                        panic!("expected test name");
                    };
                    assert!(
                        runnable
                            .insert(name.path.get_ident().unwrap().to_string(), false)
                            .is_none()
                    );
                }
                _ => {}
            }
        }
        let mut mapped = BTreeSet::new();
        for test in entry["tests"].as_array().unwrap() {
            assert_eq!(
                test["status"], "active",
                "all migrated contracts must be active"
            );
            let targets = test["rust"].as_array().unwrap();
            assert!(
                !targets.is_empty(),
                "unmapped Python test: {}",
                test["python"]
            );
            for name in targets {
                let name = name.as_str().unwrap();
                assert!(
                    mapped.insert(name.to_owned()),
                    "duplicate Rust mapping: {name}"
                );
                assert_eq!(
                    runnable.get(name),
                    Some(&false),
                    "{source}: missing or ignored Rust test {name}"
                );
            }
        }
        // This Rust-only inventory guard is the sole test without a Python source.
        runnable.remove("test_port_inventory_covers_all_python_tests");
        assert_eq!(runnable.keys().cloned().collect::<BTreeSet<_>>(), mapped);
    }
    let actual_sources: BTreeSet<_> = fs::read_dir(root().join("tests_refsol"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with("test_week_4_") && name.ends_with(".py"))
        .map(|name| format!("tests_refsol/{name}"))
        .collect();
    assert_eq!(actual_sources, recorded_sources);
}
