import asyncio
import json
import logging

from typing import ClassVar

from livekit.agents import (
    AgentSession,
    ChatContext,
    ConversationItemAddedEvent,
)

logger = logging.getLogger("observer")

class PolicyObserver:
    """
    Monitors the ride-share support conversation in parallel for policy violations.

    Runs an LLM evaluation after every user turn. If an eval is already in
    flight when a new turn arrives, a follow-up eval is scheduled to run
    immediately after, so no content is ever skipped.

    When a violation is detected, injects a [POLICY: ...] system message into
    the active agent's chat_ctx via update_chat_ctx. The agent sees the hint
    on its next reply cycle without any interruption or handoff. Each violation
    type is injected at most once per call to avoid noise.
    """

    VIOLATION_KEYS: ClassVar[list[str]] = [
        "safety_emergency",  # Caller is in immediate physical danger
        "threatening_language",  # Threats toward driver, agent, or others
        "discrimination_report",  # Discriminatory driver behavior
        "fraud_attempt",  # Fabricating or manipulating a claim
        "harassment_report",  # Sexual or persistent verbal harassment
    ]

    def __init__(self, session: AgentSession, observer_llm) -> None:
        self.session = session
        self.observer_llm = observer_llm
        self.conversation_history: list[dict] = []
        self.injected_violations: set[str] = set()  # Each type injected at most once
        self._evaluating = False
        self._pending_eval = False  # New content arrived while eval was in flight
        self._bg_tasks: set[asyncio.Task] = set()  # Keep references to prevent GC

        self._setup_listeners()
        logger.info("[OBSERVER] Policy observer attached to session")

    def _setup_listeners(self) -> None:
        @self.session.on("conversation_item_added")
        def on_item_added(event: ConversationItemAddedEvent) -> None:
            # Only monitor user speech, not agent replies
            if event.item.role != "user":
                return

            text = "".join(c for c in event.item.content if isinstance(c, str))
            if not text.strip():
                return

            self.conversation_history.append({"role": "user", "text": text})
            logger.debug(f"[OBSERVER] Buffered: {text[:80]}")

            # Fire an eval on every user turn. If one is already running,
            # _pending_eval ensures a follow-up runs when it finishes.
            task = asyncio.create_task(self._evaluate())
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)

    async def _evaluate(self) -> None:
        """
        Executa a avaliação das políticas utilizando uma LLM
        sobre o histórico recente da conversa.

        Apenas uma avaliação é executada por vez.

        Se uma nova mensagem do usuário chegar enquanto uma
        avaliação estiver em andamento, a variável _pending_eval
        será definida como True.

        Quando a avaliação atual terminar, uma nova avaliação
        será agendada para processar as mensagens acumuladas.

        Isso evita chamadas simultâneas desnecessárias à LLM
        e garante que novas mensagens sejam consideradas.
        """

        # Verifica se já existe uma avaliação em execução.
        if self._evaluating:
            self._pending_eval = True
            return

        # Marca o início da avaliação.
        self._evaluating = True
        self._pending_eval = False

        try:
            # Recupera as últimas 10 mensagens do usuário.
            recent = self.conversation_history[-10:]

            # Organiza o histórico em formato de texto.
            transcript = "\n".join(
                f"cliente: {m['text']}" for m in recent
            )

            # Prompt utilizado pela LLM observadora.
            prompt = f"""
    Você é um sistema responsável por monitorar
    o cumprimento de políticas em uma central de
    atendimento de uma empresa de transporte por aplicativo.

    Analise as declarações do cliente apresentadas abaixo.

    Identifique possíveis situações de emergência,
    ameaças, discriminação, tentativas de fraude
    ou relatos de assédio.

    Retorne SOMENTE um objeto JSON válido,
    sem explicações ou textos adicionais.

    Histórico da conversa:
    {transcript}

    Retorne exatamente esta estrutura JSON:
    {{
      "safety_emergency": false,
      "threatening_language": false,
      "discrimination_report": false,
      "fraud_attempt": false,
      "harassment_report": false,
      "details": ""
    }}

    Defina como true as categorias identificadas.

    Utilize o campo "details" para apresentar
    uma breve justificativa em português.
    """

            # Cria o contexto de entrada da LLM.
            chat_ctx = ChatContext()

            chat_ctx.add_message(
                role="user",
                content=prompt
            )

            # Variável que armazenará a resposta da LLM.
            response_text = ""

            # Executa a avaliação utilizando a LLM observadora.
            async with self.observer_llm.chat(
                    chat_ctx=chat_ctx
            ) as stream:

                # Recebe os fragmentos da resposta.
                async for chunk in stream:
                    if chunk.delta and chunk.delta.content:
                        response_text += chunk.delta.content

            # Converte a resposta da LLM em um dicionário.
            result = self._parse_response(response_text)

            # Processa as possíveis violações identificadas.
            if result:
                await self._process_violations(result)

        except Exception:
            # Registra possíveis erros durante a avaliação.
            logger.exception(
                "[OBSERVER] Erro durante a avaliação das políticas"
            )

        finally:
            # Marca a avaliação como concluída.
            self._evaluating = False

            # Verifica se chegaram novas mensagens
            # durante a avaliação anterior.
            if self._pending_eval:
                logger.info(
                    "[OBSERVER] Novas mensagens detectadas. "
                    "Executando outra avaliação."
                )

                # Agenda uma nova avaliação assíncrona.
                task = asyncio.create_task(
                    self._evaluate()
                )

                # Mantém uma referência da tarefa.
                self._bg_tasks.add(task)

                # Remove a referência após a conclusão.
                task.add_done_callback(
                    self._bg_tasks.discard
                )

    def _parse_response(self, text: str) -> dict | None:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        logger.warning(f"[OBSERVER] Could not parse response: {text[:100]}")
        return None

    async def _inject_guardrail(self, violation: str, details: str) -> None:
        """Injeta uma instrução de segurança no contexto do agente ativo.

        Cria uma cópia do contexto atual da conversa, adiciona uma
        mensagem de sistema com a instrução de segurança e, em seguida,
        atualiza o contexto do agente.

        O agente receberá essa instrução no próximo ciclo de resposta,
        sem qualquer interrupção na conversa.
        """
        current_agent = self.session.current_agent
        if not current_agent:
            logger.warning("[OBSERVADOR] Nenhum agente ativo para receber a instrução de segurança")
            return

        hint = self.GUARDRAIL_HINTS.get(violation, "")
        if details:
            hint = f"{hint}\n\nAnálise do observador: {details}"

        ctx_copy = current_agent.chat_ctx.copy()
        ctx_copy.add_message(role="system", content=hint)

        await current_agent.update_chat_ctx(ctx_copy)

        logger.info(f"[OBSERVADOR] Instrução de segurança injetada: {violation}")

    async def _process_violations(self, result: dict) -> None:
        for key in self.VIOLATION_KEYS:
            if result.get(key) and key not in self.injected_violations:
                details = result.get("details", "")
                logger.warning(f"[OBSERVER] Violation: {key} — {details}")
                await self._inject_guardrail(key, details)
                self.injected_violations.add(key)

    from typing import ClassVar

    GUARDRAIL_HINTS: ClassVar[dict[str, str]] = {
        "safety_emergency": (
            "[POLICY: SAFETY EMERGENCY] "
            "O cliente pode estar em perigo físico imediato. "
            "Utilize imediatamente a ferramenta escalate_to_safety_team. "
            "Mantenha o cliente calmo e na linha."
        ),
        "threatening_language": (
            "[POLICY: THREATENING LANGUAGE] "
            "Foi identificada linguagem ameaçadora. "
            "Mantenha a calma e evite intensificar o conflito."
        ),
        "fraud_attempt": (
            "[POLICY: FRAUD FLAG] "
            "Há suspeita de fraude. "
            "Continue coletando informações sem acusar o cliente."
        ),
        "harassment_report": (
            "[POLICY: HARASSMENT REPORT] "
            "O cliente relatou assédio. "
            "Utilize file_driver_report para registrar a denúncia."
        ),
    }

