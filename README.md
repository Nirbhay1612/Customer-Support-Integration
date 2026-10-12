# AI Customer Support Integration System

An AI-powered customer-support platform that combines **LLM response generation, Retrieval-Augmented Generation (RAG), intent classification, business API integrations, database access, ticket escalation, and workflow automation** in one modular application.

The system is designed to answer knowledge-base questions with relevant context, retrieve customer and order information when configured, support ticket workflows, and escalate issues that require human assistance.

> **Project status:** The project’s automated test suite has been reported as **171 passed**. Deployment and live third-party integration behavior should still be verified in the target environment.

---

## Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configure Environment Variables](#configure-environment-variables)
- [Prepare the Knowledge Base](#prepare-the-knowledge-base)
- [Seed Sample Data](#seed-sample-data)
- [Run the Application](#run-the-application)
- [Run Tests](#run-tests)
- [Integrations](#integrations)
- [Security Notes](#security-notes)
- [Example Use Cases](#example-use-cases)
- [Potential Improvements](#potential-improvements)
- [Author](#author)
- [License](#license)

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

| Technology | Purpose |
|---|---|
| Python | Core application language |
| Streamlit | User interface |
| Groq / LLM provider | Language-model response generation |
| LangChain | AI/RAG orchestration where used |
| Sentence Transformers | Text embeddings |
| FAISS | Vector similarity search |
| SQLAlchemy | Database layer |
| SQLite / PostgreSQL | Data storage options |
| Requests | External API communication |
| pytest | Automated testing |
| python-dotenv | Environment-variable loading |

The exact dependencies and versions are defined in `requirements.txt`.

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

## Getting Started

### Prerequisites

- Python installed (Python 3.12 was used during the reported test run)
- Git
- A terminal
- A Groq API key or the LLM provider credentials used by your configuration
- Credentials and base URLs for any external APIs you intend to enable

### 1. Clone the repository

Replace `YOUR_USERNAME` with your GitHub username and confirm the repository URL.

```bash
git clone https://github.com/YOUR_USERNAME/ai-customer-support-integration.git
cd ai-customer-support-integration
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use your environment's approved activation method or activate the environment from your IDE.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example environment file:

**Windows PowerShell**

```powershell
Copy-Item .env.example .env
```

**macOS / Linux**

```bash
cp .env.example .env
```

Add the values required by `config/settings.py`. At minimum, the LLM integration may require:

```env
GROQ_API_KEY=replace_with_your_key
```

The exact names for database, SMTP, customer API, order API, ticket API, and webhook settings depend on the application's settings module and enabled integrations. Use `.env.example` as the source of truth; do not add credentials to source code.

## Configure Environment Variables

Review `config/settings.py` and `.env.example` to identify every required setting. Typical integration categories include:

| Category | Example setting | Purpose |
|---|---|---|
| LLM | `GROQ_API_KEY` | Authenticates the LLM client |
| Customer API | Project-specific base URL / API key | Retrieves customer data |
| Order API | Project-specific base URL / API key | Retrieves order information |
| Ticket API | Project-specific base URL / API key | Creates and updates tickets |
| SMTP | Host, port, username, password | Sends support notifications |
| Webhooks | `WEBHOOK_SECRET` | Verifies webhook signatures |
| Database | Project-specific database URL | Configures persistence |

These are configuration categories, not a guarantee that every variable uses the exact names shown. Confirm the required variable names in the code before running the app.

## Prepare the Knowledge Base

1. Add or edit support documents in `data/knowledge_base/`.
2. Keep policy details accurate and up to date.
3. Ingest the documents:

```bash
python scripts/ingest_documents.py
```

The ingestion script is intended to process the documents and create/update the vector store used by retrieval. Check its output and the configuration before relying on the index.

## Seed Sample Data

To load sample customer and order data, if supported by your current database configuration, run:

```bash
python scripts/seed_database.py
```

Use sample or synthetic records during development. Do not place real customer personal data in a public repository.

## Run the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

Open the local URL printed in the terminal. The available features depend on your environment configuration and the integrations that are reachable.

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

- GitHub: `https://github.com/YOUR_USERNAME`
- LinkedIn: `https://linkedin.com/in/YOUR_USERNAME`

Replace these placeholders with your real public profile URLs before publishing.

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.
