# AI Customer Support Integration System

An AI-powered customer-support platform that combines **LLM response generation, Retrieval-Augmented Generation (RAG), intent classification, business API integrations, database access, ticket escalation, and workflow automation** in one modular application.

The system is designed to answer knowledge-base questions with relevant context, retrieve customer and order information when configured, support ticket workflows, and escalate issues that require human assistance.

> **Project status:** The project’s automated test suite has been reported as **171 passed**. Deployment and live third-party integration behavior should still be verified in the target environment.

---

# Demo Link
[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-support-integration-e46pvet7ku5rdizvf6nljv.streamlit.app/)


## Key Features

- **AI customer-support chatbot** with a Streamlit interface
- **RAG pipeline** for knowledge-grounded responses
- Semantic search over support documents
- Customer intent classification and workflow routing
- Customer and order information lookup through API clients
- Return, refund, shipping, delivery, and payment support flows
- Support-ticket creation and ticket updates
- Email notifications for support events
- Webhook signature verification and event parsing
- Human-agent escalation for issues that need additional support
- Conversation-context management
- Database models and repository layer
- Structured logging, input validation, and exception handling
- Automated tests with pytest

## Architecture

```text
Customer
   |
   v
Streamlit UI (app.py)
   |
   v
Customer Support Chatbot
   |
   +-------------------+--------------------+
   |                   |                    |
   v                   v                    v
Intent Classifier   RAG Pipeline       Support Workflows
   |                   |                    |
   v                   v                    v
LLM / Prompts       Retriever          Business Integrations
                       |                    |
                       v           +--------+---------+
                  FAISS Vector     |        |         |
                     Store         v        v         v
                               Customer   Order    Ticket / Email
                                  API      API       Services

                         Database / Repository
```

The exact runtime path depends on the customer intent and the integrations configured in the environment.

## How It Works

1. **Customer query:** A customer submits a question through the Streamlit interface.
2. **Intent classification:** The application identifies the likely intent, such as an order-status question or a return request.
3. **Workflow routing:** The support workflow determines whether to retrieve knowledge, call a business integration, or escalate the request.
4. **Knowledge retrieval:** For knowledge-based questions, the RAG pipeline loads and chunks documents, creates embeddings, and searches the FAISS vector store.
5. **Response generation:** The LLM generates a response using the available context and conversation history.
6. **Escalation:** If the issue requires human assistance, the system can use ticket and notification integrations when configured.

## RAG Pipeline

```text
Knowledge-base documents
        |
        v
Document loader
        |
        v
Text chunking
        |
        v
Embeddings
        |
        v
FAISS vector store
        |
        v
Retriever
        |
        v
Relevant context
        |
        v
LLM response generation
```

The sample knowledge base is organized around FAQs, shipping, returns, refunds, and payments.

## Technology Stack

| Technology | Badge |
|---|---|
| Python | [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) |
| Streamlit | [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/) |
| Groq | [![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/) |
| LangChain | [![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/) |
| Sentence Transformers | [![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://sbert.net/) |
| FAISS | [![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF?style=for-the-badge)](https://github.com/facebookresearch/faiss) |
| SQLAlchemy | [![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=for-the-badge&logo=sqlalchemy&logoColor=white)](https://www.sqlalchemy.org/) |
| SQLite | [![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/) |
| PostgreSQL | [![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/) |
| Requests | [![Requests](https://img.shields.io/badge/Requests-2CA5E0?style=for-the-badge)](https://requests.readthedocs.io/) |
| pytest | [![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org/) |
| python-dotenv | [![python-dotenv](https://img.shields.io/badge/python--dotenv-000000?style=for-the-badge)](https://github.com/theskumar/python-dotenv) |

## Project Structure

```text
ai-customer-support-integration/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── LICENSE
├── config/
│   ├── __init__.py
│   └── settings.py
├── src/
│   ├── __init__.py
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── llm.py
│   │   ├── prompts.py
│   │   ├── intent_classifier.py
│   │   └── response_generator.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── document_loader.py
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── retriever.py
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── customer_api.py
│   │   ├── order_api.py
│   │   ├── email_service.py
│   │   ├── ticket_service.py
│   │   └── webhook.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py
│   │   ├── models.py
│   │   └── repository.py
│   ├── workflows/
│   │   ├── __init__.py
│   │   ├── support_workflow.py
│   │   ├── escalation.py
│   │   └── automation.py
│   ├── chatbot/
│   │   ├── __init__.py
│   │   ├── chatbot.py
│   │   ├── context_manager.py
│   │   └── response_handler.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── validators.py
│       └── exceptions.py
├── data/
│   ├── knowledge_base/
│   │   ├── faq.md
│   │   ├── shipping.md
│   │   ├── returns.md
│   │   ├── refunds.md
│   │   └── payments.md
│   └── sample_data/
│       ├── customers.json
│       └── orders.json
├── tests/
│   ├── test_ai.py
│   ├── test_rag.py
│   ├── test_integrations.py
│   └── test_workflows.py
├── scripts/
│   ├── ingest_documents.py
│   └── seed_database.py
└── docs/
    ├── architecture.md
    └── api_integration.md
```

> This tree reflects the intended project layout. If your local repository differs, update this section to match the actual files.



## Run Tests

Run the complete test suite:

```bash
python -m pytest -v
```

Run individual test modules when debugging:

```bash
python -m pytest tests/test_ai.py -v
python -m pytest tests/test_rag.py -v
python -m pytest tests/test_integrations.py -v
python -m pytest tests/test_workflows.py -v
```

**Reported test status:** 171 tests passed in the latest run shared during project development. Re-run the suite after making changes and before deploying; results may differ by code version and environment.

## Integrations

### Customer API

Used to retrieve customer information through a configured API client.

### Order API

Supports order lookups such as order status, delivery information, and order details, depending on the connected API's capabilities.

### Ticket Service

Provides ticket creation, retrieval, and update operations for support workflows.

### Email Service

Sends customer notifications and support updates through configured SMTP settings.

### Webhooks

Verifies HMAC-SHA256 signatures and parses incoming event payloads. Keep the webhook secret private and configure the sender to use the expected signature format.

> External integrations require valid configuration and reachable services. Unit tests alone do not establish that a live third-party service is configured or operational.

## Security Notes

- Store credentials in environment variables or an appropriate secrets manager.
- Keep `.env` out of version control.
- Never commit API keys, SMTP passwords, tokens, webhook secrets, or production data.
- Use synthetic data for examples and tests.
- Validate user input and external API responses.
- Keep logs free of credentials and unnecessary personal data.
- Use HTTPS/TLS for external communication.
- Restrict access to any deployed application and administrative endpoints.
- Confirm authorization before exposing customer or order data.
- Verify webhook signatures before processing events.

Before publishing, inspect the repository for secrets and confirm `.gitignore` excludes local environment files, virtual environments, caches, and generated sensitive data.

## Example Use Cases

- E-commerce customer support
- SaaS product support
- Online marketplace assistance
- Logistics and delivery inquiries
- Subscription support
- Internal company help desks

These are potential applications of the architecture; each use case needs appropriate business rules, data access, and security controls.

## Potential Improvements

Possible future enhancements include:

- Authentication and role-based authorization
- Human-agent dashboard
- Admin dashboard and support analytics
- PostgreSQL-backed production persistence
- Redis-based conversation memory
- CRM, WhatsApp, or Slack integrations
- Voice-based support
- Response-quality evaluation and feedback
- Observability, tracing, and metrics
- Docker and cloud deployment automation

## Author

**Your Name**  
AI / Automation Developer



## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.
