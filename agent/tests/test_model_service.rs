use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::thread;
use std::time::Duration;

use serde_json::{Value, json};
use tiny_llm_agent::generation::{Generate, Message};
use tiny_llm_agent::model_service::HttpGenerator;

fn mock_service(
    replies: Vec<(u16, String)>,
    delay: Duration,
) -> (String, thread::JoinHandle<Vec<Value>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/", listener.local_addr().unwrap());
    let handle = thread::spawn(move || {
        let mut requests = Vec::new();
        for (status, body) in replies {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut reader = BufReader::new(&mut stream);
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            assert_eq!(line.trim(), "POST /generate HTTP/1.1");
            let mut length = None;
            loop {
                line.clear();
                assert!(reader.read_line(&mut line).unwrap() > 0);
                if line == "\r\n" {
                    break;
                }
                if let Some((key, value)) = line.split_once(':')
                    && key.eq_ignore_ascii_case("content-length")
                {
                    length = Some(value.trim().parse::<usize>().unwrap());
                }
            }
            let mut bytes = vec![0; length.unwrap()];
            reader.read_exact(&mut bytes).unwrap();
            requests.push(serde_json::from_slice(&bytes).unwrap());
            thread::sleep(delay);
            // The timeout test intentionally closes the client before this write.
            let _ = write!(
                stream,
                "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        }
        requests
    });
    (url, handle)
}

fn message(role: &str, content: &str) -> Message {
    Message::from([
        ("role".into(), role.into()),
        ("content".into(), content.into()),
    ])
}

#[test]
fn sends_complete_messages_on_each_call_and_preserves_raw_response() {
    let raw = "  <think>思考</think>\n{\"final\":\"完成\"}\n";
    let body = json!({"response": raw}).to_string();
    let (url, server) = mock_service(vec![(200, body.clone()), (200, body)], Duration::ZERO);
    let mut generator = HttpGenerator::new(&url, 42, true, Duration::from_secs(5)).unwrap();
    let mut messages = vec![
        message("system", "instructions"),
        message("user", "检查文件"),
    ];
    assert_eq!(generator.generate(&messages).unwrap(), raw);
    messages.push(message(
        "assistant",
        r#"{"tool":"read_file","path":"README.md"}"#,
    ));
    messages.push(message("user", "Tool result:\nfile contents"));
    assert_eq!(generator.generate(&messages).unwrap(), raw);
    let requests = server.join().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0]["messages"], json!(&messages[..2]));
    assert_eq!(
        requests[1],
        json!({"messages": messages, "max_tokens": 42, "enable_thinking": true})
    );
}

#[test]
fn reports_service_and_protocol_errors_without_retrying() {
    for (status, body, expected) in [
        (
            500,
            r#"{"error":"generation exploded"}"#,
            "generation exploded",
        ),
        (400, r#"{"error":"bad messages"}"#, "HTTP 400"),
        (302, "redirect", "HTTP 302"),
        (200, "not json", "invalid model service JSON"),
        (200, r#"{"response":12}"#, "string 'response'"),
        (200, r#"{}"#, "string 'response'"),
    ] {
        let (url, server) = mock_service(vec![(status, body.into())], Duration::ZERO);
        let mut generator = HttpGenerator::new(&url, 256, false, Duration::from_secs(5)).unwrap();
        let error = generator.generate(&[message("user", "task")]).unwrap_err();
        assert!(error.0.contains(expected), "{error}");
        assert_eq!(server.join().unwrap().len(), 1);
    }
}

#[test]
fn reports_connection_failure() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let mut generator = HttpGenerator::new(&url, 256, false, Duration::from_secs(1)).unwrap();
    assert!(
        generator
            .generate(&[message("user", "task")])
            .unwrap_err()
            .0
            .contains("request failed")
    );
}

#[test]
fn times_out_slow_generation() {
    let (url, server) = mock_service(
        vec![(200, r#"{"response":"late"}"#.into())],
        Duration::from_millis(250),
    );
    let mut generator = HttpGenerator::new(&url, 256, false, Duration::from_millis(50)).unwrap();
    assert!(generator.generate(&[message("user", "task")]).is_err());
    assert_eq!(server.join().unwrap().len(), 1);
}

#[test]
fn validates_client_configuration() {
    for url in [
        "garbage",
        concat!("file://", env!("CARGO_MANIFEST_DIR"), "/tmp/model"),
        "http://localhost/?token=x",
        "http://localhost/#fragment",
    ] {
        assert!(HttpGenerator::new(url, 256, false, Duration::from_secs(1)).is_err());
    }
    assert!(HttpGenerator::new("http://localhost:8000", 0, false, Duration::from_secs(1)).is_err());
    assert!(HttpGenerator::new("http://localhost:8000", 256, false, Duration::ZERO).is_err());
}
