import os
import sys
import json
import requests

def run_tests():
    base_url = "http://localhost:8080"
    print("==================================================")
    print("Iniciando testes E2E do Projeto Predictive Log Platform")
    print(f"URL Alvo: {base_url}")
    print("==================================================")

    # --- TESTE 1: Ingestão de Logs (Upload de CSV) ---
    print("\n[Teste 1/3] Testando Ingestão de Logs (/logs/upload)...")
    csv_content = (
        "timestamp,method,path,status_code,response_time_ms,user_agent,ip_address,bytes_sent\n"
        "2026-06-07T10:00:00,GET,/api/data,200,15.5,Mozilla/5.0,192.168.1.1,512\n"
        "2026-06-07T10:05:00,POST,/api/users,201,120.2,curl/7.81.0,10.0.0.5,1024\n"
        "2026-06-07T10:10:00,GET,/api/faulty,500,350.0,Mozilla/5.0,192.168.1.1,0\n"
    )

    try:
        files = {'file': ('logs.csv', csv_content, 'text/csv')}
        response = requests.post(f"{base_url}/logs/upload", files=files)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code != 200:
            print(f"Erro: Resposta inesperada: {response.text}")
            sys.exit(1)
            
        data = response.json()
        print("Resposta do upload de log:")
        print(json.dumps(data, indent=2))
        
        assert data.get("status") == "success", "Status de resposta deveria ser 'success'"
        assert data.get("recordsProcessed") == 3, f"Registros processados deveria ser 3, veio {data.get('recordsProcessed')}"
        print("✔ Teste 1/3 concluído com SUCESSO!")
    except Exception as e:
        print(f"❌ Teste 1/3 falhou: {str(e)}")
        sys.exit(1)

    # --- TESTE 2: Predição de Erro ---
    print("\n[Teste 2/3] Testando Predição de Erro (/predict/error)...")
    payload = {
        "method": "GET",
        "hour": 10,
        "historicalAvgResponse": 150.0,
        "dayOfWeek": 1
    }

    try:
        response = requests.post(f"{base_url}/predict/error", json=payload)
        print(f"Status Code: {response.status_code}")
        if response.status_code != 200:
            print(f"Erro: Resposta inesperada: {response.text}")
            sys.exit(1)

        data = response.json()
        print("Resposta da predição de erro:")
        print(json.dumps(data, indent=2))

        assert "error_probability" in data, "Falta o campo 'error_probability' na resposta"
        assert "risk_level" in data, "Falta o campo 'risk_level' na resposta"
        assert "model_used" in data, "Falta o campo 'model_used' na resposta"
        assert "inference_time_ms" in data, "Falta o campo 'inference_time_ms' na resposta"
        print("✔ Teste 2/3 concluído com SUCESSO!")
    except Exception as e:
        print(f"❌ Teste 2/3 falhou: {str(e)}")
        sys.exit(1)

    # --- TESTE 3: Predição de Tempo de Resposta ---
    print("\n[Teste 3/3] Testando Predição de Tempo de Resposta (/predict/response-time)...")
    try:
        response = requests.post(f"{base_url}/predict/response-time", json=payload)
        print(f"Status Code: {response.status_code}")
        if response.status_code != 200:
            print(f"Erro: Resposta inesperada: {response.text}")
            sys.exit(1)

        data = response.json()
        print("Resposta da predição de tempo de resposta:")
        print(json.dumps(data, indent=2))

        assert "predicted_response_time_ms" in data, "Falta o campo 'predicted_response_time_ms' na resposta"
        assert "confidence_interval" in data, "Falta o campo 'confidence_interval' na resposta"
        assert "model_used" in data, "Falta o campo 'model_used' na resposta"
        assert "inference_time_ms" in data, "Falta o campo 'inference_time_ms' na resposta"
        
        ci = data["confidence_interval"]
        assert "lower_bound_ms" in ci, "Falta 'lower_bound_ms' no confidence_interval"
        assert "upper_bound_ms" in ci, "Falta 'upper_bound_ms' no confidence_interval"
        assert "confidence_level" in ci, "Falta 'confidence_level' no confidence_interval"
        
        print("✔ Teste 3/3 concluído com SUCESSO!")
    except Exception as e:
        print(f"❌ Teste 3/3 falhou: {str(e)}")
        sys.exit(1)

    print("\n==================================================")
    print("TODOS OS TESTES E2E PASSARAM COM SUCESSO!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
