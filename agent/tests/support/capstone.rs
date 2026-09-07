//! Supplied deterministic scenario, mirroring week4-capstone.py using the Rust learner API.
//! This is orchestration and a scripted model, not implementations of course exercises.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use tiny_llm_agent::generation::{Generate, Message};
use tiny_llm_agent::workspace::ConfirmResult;
use tiny_llm_agent::*;

const TASK: &str = "Set app.py to answer = 2, validate it, and inspect the build evidence.";
const DIAGNOSTIC: &str = "ERROR code=E42 dependency mismatch";

#[derive(Clone)]
struct ScriptedModel {
    responses: Vec<String>,
    index: usize,
    checkpoint: Option<ModelCheckpoint>,
    prefix: Vec<Message>,
    restored: bool,
}
impl ScriptedModel {
    fn new(responses: Vec<String>) -> Self {
        Self {
            responses,
            index: 0,
            checkpoint: None,
            prefix: vec![],
            restored: false,
        }
    }
    fn fork(&self, responses: Vec<String>) -> Self {
        assert!(self.checkpoint.is_some());
        Self {
            responses,
            ..self.clone()
        }
    }
}
impl Generate for ScriptedModel {
    fn generate(&mut self, messages: &[Message]) -> Result<String, AgentError> {
        if self.checkpoint.is_some() {
            assert!(self.restored, "restore the checkpoint before generating");
            assert!(
                messages.starts_with(&self.prefix),
                "preserve the saved prefix"
            );
        }
        let response = self
            .responses
            .get(self.index)
            .expect("scripted model exhausted")
            .clone();
        self.index += 1;
        Ok(response)
    }
    fn save_checkpoint(&mut self, messages: &[Message]) -> Result<ModelCheckpoint, AgentError> {
        let tokens = messages
            .iter()
            .map(|m| m["content"].chars().count() as i64)
            .collect();
        let model = ModelCheckpoint::new(
            messages.len() as i64,
            self.index as i64,
            tokens,
            vec![messages.len() as i64; 2],
        )?;
        self.checkpoint = Some(model.clone());
        self.prefix = messages.to_vec();
        self.restored = false;
        Ok(model)
    }
    fn restore_checkpoint(&mut self, checkpoint: &ModelCheckpoint) -> Result<(), AgentError> {
        assert_eq!(Some(checkpoint), self.checkpoint.as_ref());
        self.index = checkpoint.response_index as usize;
        self.restored = true;
        Ok(())
    }
    fn prefix_reuse(&self) -> Option<(i64, Vec<i64>, i64)> {
        self.checkpoint.as_ref().map(|c| {
            (
                c.cached_token_ids.len() as i64,
                c.layer_offsets.clone(),
                c.cached_token_ids.len() as i64,
            )
        })
    }
}
fn action(value: Value) -> String {
    serde_json::to_string(&value).unwrap()
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn receipt_ids(store: &ReceiptStore) -> Vec<String> {
    (1..16)
        .map_while(|i| store.get(&format!("call-{i}")).map(|r| r.receipt_id()))
        .collect()
}
fn workspace(root: &Path, command: &[String], approve: bool) -> Workspace {
    Workspace::new(
        ToolPolicy::new(
            root.join("workspace"),
            65536,
            200,
            true,
            vec![command.to_vec()],
            65536,
            30.0,
        )
        .unwrap(),
        Some(Box::new(move |_| {
            if approve {
                ConfirmResult::Approved(true)
            } else {
                ConfirmResult::Decision(
                    ApprovalDecision::new(false, "keep the requested answer at 2".into()).unwrap(),
                )
            }
        })),
        ReceiptStore::new(Some(root.join("receipts.jsonl"))).unwrap(),
    )
}
fn branch_record(outcome: &BranchOutcome) -> Value {
    json!({"name": outcome.name, "steering": outcome.steering,
        "evaluation": {"passed": outcome.report.passed(), "checks": outcome.report.checks.iter().map(|c| json!({"name": c.name, "passed": c.passed, "detail": c.detail})).collect::<Vec<_>>()},
        "reused_tokens": outcome.reuse.reused_tokens, "layer_offsets": outcome.reuse.layer_offsets,
        "avoided_prefill_tokens": outcome.reuse.avoided_prefill_tokens})
}
fn payload(text: &str, prefix: &str) -> Value {
    serde_json::from_str(text.strip_prefix(prefix).expect("observation prefix")).unwrap()
}

pub fn run_capstone() -> Value {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().join("base");
    fs::create_dir_all(base.join("workspace")).unwrap();
    let build_log = format!(
        "build start α\n{}{DIAGNOSTIC}\n{}build end\n",
        "compile-unit-ok\n".repeat(96),
        "link-unit-ok\n".repeat(96)
    );
    fs::write(base.join("workspace/app.py"), "answer = 1\n").unwrap();
    fs::write(base.join("workspace/build-source.txt"), &build_log).unwrap();
    // Fixed argv keeps executable/temp paths out of receipt identities.
    let command: Vec<String> = ["/bin/sh", "-c", "printf 'answer = 2\\n' | cmp -s - app.py && cp build-source.txt build.log && printf 'validation passed\\n' && cat build.log"].map(str::to_owned).to_vec();
    let mut base_workspace = workspace(&base, &command, true);
    let mut base_model = ScriptedModel::new(vec![
        action(json!({"tool":"read_file","path":"app.py"})),
        action(json!({"tool":"edit_file","path":"app.py","old":"1","new":"2"})),
        action(json!({"tool":"run_command","argv":command})),
        action(json!({"tool":"read_file","path":"build.log"})),
    ]);
    let limits = AgentLimits::new(4, 48000, 3, 2).unwrap();
    let checkpoint =
        run_to_checkpoint(TASK, &mut base_model, &mut base_workspace, 4, Some(&limits)).unwrap();
    assert_eq!(base_workspace.modified_files(), ["app.py"]);
    let receipts: Vec<_> = ["call-1", "call-2"]
        .iter()
        .map(|id| base_workspace.receipt_store.get(id).unwrap().clone())
        .collect();
    let messages = checkpoint
        .messages
        .iter()
        .map(|(role, content)| {
            Message::from([
                ("role".into(), role.clone()),
                ("content".into(), content.clone()),
            ])
        })
        .collect::<Vec<_>>();
    let compact = compact_completed_interactions(
        &messages,
        &receipts,
        &|ms| ms.iter().map(|m| m["content"].chars().count() as i64).sum(),
        0,
        80,
    )
    .unwrap();
    assert_eq!(compact.compacted_interactions, 2);
    assert!(compact.saved_tokens() > 0);
    let mut prefix = ScriptedModel::new(vec![]);
    let model = prefix.save_checkpoint(&compact.messages).unwrap();
    let checkpoint = create_checkpoint(TASK, &compact.messages, model).unwrap();
    let status = inspect_checkpoint(&checkpoint, 160).unwrap();
    let initial_receipts = fs::read(base.join("receipts.jsonl")).unwrap();
    for name in ["validate-only", "try-extra-edit"] {
        let root = temp.path().join(name);
        fs::create_dir_all(root.join("workspace")).unwrap();
        for file in ["app.py", "build-source.txt", "build.log"] {
            fs::copy(
                base.join("workspace").join(file),
                root.join("workspace").join(file),
            )
            .unwrap();
        }
        fs::copy(base.join("receipts.jsonl"), root.join("receipts.jsonl")).unwrap();
        assert_eq!(
            fs::read(root.join("receipts.jsonl")).unwrap(),
            initial_receipts
        );
    }
    let passing_root = temp.path().join("validate-only");
    let failing_root = temp.path().join("try-extra-edit");
    let mut passing_workspace = workspace(&passing_root, &command, true);
    let mut failing_workspace = workspace(&failing_root, &command, false);
    let mut passing_receipts =
        ReceiptStore::new(Some(passing_root.join("receipts.jsonl"))).unwrap();
    let mut failing_receipts =
        ReceiptStore::new(Some(failing_root.join("receipts.jsonl"))).unwrap();
    let case = EvaluationCase::new(
        "validated branch".into(),
        vec![FileExpectation::new("app.py".into(), "answer = 2\n".into()).unwrap()],
        vec![ResultExpectation::new("run_command".into(), "validation passed".into()).unwrap()],
        vec![
            ReceiptExpectation::new(
                "call-1".into(),
                "edit_file".into(),
                "ok".into(),
                "edited app.py".into(),
                vec!["app.py".into()],
            )
            .unwrap(),
            ReceiptExpectation::new(
                "call-2".into(),
                "run_command".into(),
                "ok".into(),
                "validation passed".into(),
                vec![],
            )
            .unwrap(),
            ReceiptExpectation::new(
                "call-3".into(),
                "run_command".into(),
                "ok".into(),
                "validation passed".into(),
                vec![],
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let mut passing_model = prefix.fork(vec![
        action(json!({"tool":"run_command","argv":command})),
        action(json!({"final":"validated branch"})),
    ]);
    let mut failing_model = prefix.fork(vec![
        action(json!({"tool":"read_file","path":"app.py"})),
        action(json!({"tool":"edit_file","path":"app.py","old":"2","new":"3"})),
        action(json!({"final":"extra edit denied"})),
    ]);
    let passing = run_branch(
        "validate-only",
        "validate without another edit",
        &checkpoint,
        &mut passing_model,
        &mut passing_workspace,
        &mut passing_receipts,
        &case,
        None,
    )
    .unwrap();
    let failing = run_branch(
        "try-extra-edit",
        "try changing the answer again",
        &checkpoint,
        &mut failing_model,
        &mut failing_workspace,
        &mut failing_receipts,
        &case,
        None,
    )
    .unwrap();
    assert!(passing.report.passed());
    assert!(!failing.report.passed());
    let selected = select_branch(&[passing.clone(), failing.clone()], "validate-only").unwrap();
    assert!(selected.run.events.iter().all(
        |e| !matches!(&e.action, Some(protocol::AgentAction::Tool(a)) if a.tool == "edit_file")
    ));
    assert!(
        fs::read(passing_root.join("receipts.jsonl"))
            .unwrap()
            .starts_with(&initial_receipts)
    );
    assert_eq!(
        fs::read(failing_root.join("receipts.jsonl")).unwrap(),
        initial_receipts
    );
    assert_eq!(
        fs::read(base.join("receipts.jsonl")).unwrap(),
        initial_receipts
    );
    let selected_receipt_ids = receipt_ids(&passing_workspace.receipt_store);
    let selected_modified = passing_workspace.modified_files();
    let artifact_root = temp.path().join("artifacts");
    fs::create_dir(&artifact_root).unwrap();
    let artifacts = ArtifactStore::new(artifact_root).unwrap();
    let artifact_id = format!("artifact-{}", digest(build_log.as_bytes()));
    let start = build_log.find(DIAGNOSTIC).unwrap() as i64;
    let end = start + DIAGNOSTIC.len() as i64;
    let range_path = artifacts.range_path(&artifact_id, start, end).unwrap();
    let mut bounded =
        BoundedEvidenceWorkspace::new(passing_workspace, artifacts, 512, 32, 128).unwrap();
    let mut responses = vec![
        action(json!({"tool":"read_file","path":"build.log"})),
        action(json!({"tool":"read_file","path":range_path})),
        action(json!({"final":"retrieved diagnostic E42"})),
    ]
    .into_iter();
    let mut generate = |_: &[Message]| responses.next().unwrap();
    let limits = AgentLimits::new(3, 48000, 3, 2).unwrap();
    let run = run_agent(
        Some("Inspect the selected build evidence and retrieve diagnostic E42."),
        &mut generate,
        &mut bounded,
        Some(&limits),
        None,
    )
    .unwrap();
    assert!(run.completed);
    assert_eq!(run.events.len(), 3);
    let observation = run.events[0].result.as_ref().unwrap();
    let artifact = payload(observation, "Tool result externalized:\n");
    let range = payload(run.events[1].result.as_ref().unwrap(), "Artifact range:\n");
    assert_eq!(range["artifact_id"], artifact_id);
    assert_eq!(range["start"], start);
    assert_eq!(range["end"], end);
    assert_eq!(range["data"], DIAGNOSTIC);
    json!({
        "compaction": {"tokens_before":compact.tokens_before, "tokens_after":compact.tokens_after, "saved_tokens":compact.saved_tokens(), "receipt_ids":compact.receipt_ids,
            "checkpoint_status":{"task":status.task,"last_action":status.last_action,"last_evidence":status.last_evidence,"next_step":status.next_step}},
        "branches":[branch_record(&passing),branch_record(&failing)],
        "selection":{"selected_name":selected.name,"app_py_sha256":digest(&fs::read(passing_root.join("workspace/app.py")).unwrap()),"modified_files":{"base":base_workspace.modified_files(),"selected_branch":selected_modified},"base_receipt_ids":receipt_ids(&base_workspace.receipt_store),"selected_receipt_ids":selected_receipt_ids},
        "artifact":{"artifact_id":artifact["artifact_id"],"sha256":artifact["sha256"],"full_byte_count":artifact["byte_count"],"model_visible_observation_byte_count":observation.len(),"omitted_interval":artifact["omitted_range"],"range_start":range["start"],"range_end":range["end"],"range_byte_count":range["byte_count"],"range_text":range["data"]}
    })
}
