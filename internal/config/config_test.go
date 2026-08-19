package config

import "testing"

func TestFromEnvLoadsMCPServers(t *testing.T) {
	t.Setenv("GATEWAY_API_KEYS", "gateway-key")
	t.Setenv("GEMINI_API_KEY", "gemini-key")
	t.Setenv("MCP_SERVERS", "kube=http://mcp-kube.ai.svc.cluster.local/mcp,filesystem=http://fs/mcp")
	t.Setenv("MCP_BEARER_TOKENS", "kube=kube-token")
	t.Setenv("MCP_ALLOWED_TOOLS", "kube__get*")
	t.Setenv("MCP_DENIED_TOOLS", "kube__get_secret*")
	t.Setenv("MCP_MAX_TOOL_ROUNDS", "3")
	t.Setenv("MCP_TOOL_TIMEOUT_SECONDS", "7")

	cfg, err := FromEnv()
	if err != nil {
		t.Fatal(err)
	}

	if len(cfg.MCPServers) != 2 {
		t.Fatalf("expected 2 MCP servers, got %d", len(cfg.MCPServers))
	}
	if cfg.MCPServers[0].Name != "kube" || cfg.MCPServers[0].BearerToken != "kube-token" {
		t.Fatalf("unexpected first MCP server: %#v", cfg.MCPServers[0])
	}
	if cfg.MCPMaxToolRounds != 3 || cfg.MCPToolTimeout.Seconds() != 7 {
		t.Fatalf("unexpected MCP limits: %#v", cfg)
	}
	if len(cfg.MCPAllowedTools) != 1 || cfg.MCPAllowedTools[0] != "kube__get*" {
		t.Fatalf("unexpected allowed tools: %#v", cfg.MCPAllowedTools)
	}
	if len(cfg.MCPDeniedTools) != 1 || cfg.MCPDeniedTools[0] != "kube__get_secret*" {
		t.Fatalf("unexpected denied tools: %#v", cfg.MCPDeniedTools)
	}
}

func TestFromEnvRejectsInvalidMCPServer(t *testing.T) {
	t.Setenv("GATEWAY_API_KEYS", "gateway-key")
	t.Setenv("GEMINI_API_KEY", "gemini-key")
	t.Setenv("MCP_SERVERS", "kube")

	_, err := FromEnv()
	if err == nil {
		t.Fatal("expected error")
	}
}
