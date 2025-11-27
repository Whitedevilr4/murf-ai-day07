from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# LiveKit base imports
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
)

# Tools
from livekit.agents.llm import function_tool
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

VOICE_SDR = "en-US-matthew"


# -----------------------------------------------------------
# JSON PATHS
# -----------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = BASE_DIR.parent / "shared-data" / "day7_catalog.json"
ORDERS_PATH = BASE_DIR.parent / "shared-data" / "orders.json"

CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)

if not ORDERS_PATH.exists():
    with open(ORDERS_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def load_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_orders():
    try:
        with open(ORDERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def save_orders(orders):
    with open(ORDERS_PATH, "w", encoding="utf-8") as f:
        json.dump(orders, f, indent=2)

def find_item(name, catalog):
    name = name.lower().strip()
    for item in catalog["items"]:
        if name in item["name"].lower():
            return item
    return None

def next_order_id(orders):
    if not orders:
        return "ORD-1"
    try:
        last = int(orders[-1]["order_id"].split("-")[1])
        return f"ORD-{last+1}"
    except:
        return f"ORD-{len(orders)+1}"


# -----------------------------------------------------------
# Recipes
# -----------------------------------------------------------

RECIPES = {
    "peanut butter sandwich": {
        "items": [
            {"name": "Whole Wheat Bread", "quantity": 1},
            {"name": "Peanut Butter (Crunchy)", "quantity": 1},
        ]
    },

    "pasta for two": {
        "items": [
            {"name": "Penne Pasta", "quantity": 1},
            {"name": "Tomato Basil Pasta Sauce", "quantity": 1},
        ]
    }
}


# -----------------------------------------------------------
# Grocery Agent
# -----------------------------------------------------------

class GroceryAgent(Agent):

    def __init__(self):
        super().__init__(
            instructions="""
You are QuickKart's friendly grocery ordering voice assistant.

You can:
- Add grocery items
- Remove/update items
- Add recipe ingredients
- Show cart
- Place order (save JSON)
- Track order status

Rules:
- ALWAYS use tools for cart/order operations.
- Keep tone friendly, short, conversational Indian English.
- If user says: "done", "place order", "that's all" → call finish_order.
"""
        )

    # ---------------------------------------------------
    # Tool: Add item
    # ---------------------------------------------------
    @function_tool
    async def add_item(self, item_name: str, quantity: float = 1):
        session = self.session
        catalog = load_catalog()
        cart = session.userdata.get("cart", [])

        item = find_item(item_name, catalog)
        if not item:
            return f"I couldn't find {item_name} in the store."

        for c in cart:
            if c["item_id"] == item["id"]:
                c["quantity"] += quantity
                break
        else:
            cart.append({
                "item_id": item["id"],
                "name": item["name"],
                "quantity": quantity,
                "unit_price": item["price"]
            })

        session.userdata["cart"] = cart
        return f"Added {quantity} x {item['name']}."

    # ---------------------------------------------------
    # Tool: Update quantity
    # ---------------------------------------------------
    @function_tool
    async def update_quantity(self, item_name: str, quantity: float):
        session = self.session
        cart = session.userdata.get("cart", [])

        for c in cart:
            if item_name.lower() in c["name"].lower():
                if quantity <= 0:
                    cart.remove(c)
                    session.userdata["cart"] = cart
                    return f"Removed {c['name']}."
                c["quantity"] = quantity
                session.userdata["cart"] = cart
                return f"Updated {c['name']} to {quantity}."

        return "Item not found in cart."

    # ---------------------------------------------------
    # Tool: Remove item
    # ---------------------------------------------------
    @function_tool
    async def remove_item(self, item_name: str):
        session = self.session
        cart = session.userdata.get("cart", [])

        for c in cart:
            if item_name.lower() in c["name"].lower():
                cart.remove(c)
                session.userdata["cart"] = cart
                return f"Removed {c['name']}."

        return "Item not found."

    # ---------------------------------------------------
    # Tool: Show cart
    # ---------------------------------------------------
    @function_tool
    async def show_cart(self):
        session = self.session
        cart = session.userdata.get("cart", [])

        if not cart:
            return "Your cart is empty."

        total = 0
        lines = []

        for c in cart:
            line_total = c["unit_price"] * c["quantity"]
            total += line_total
            lines.append(f"{c['quantity']} x {c['name']} = ₹{line_total}")

        lines.append(f"Total = ₹{total}")
        return "\n".join(lines)

    # ---------------------------------------------------
    # Tool: Recipe ingredients
    # ---------------------------------------------------
    @function_tool
    async def add_recipe(self, recipe: str):
        session = self.session
        catalog = load_catalog()
        cart = session.userdata.get("cart", [])

        recipe = recipe.lower().strip()
        if recipe not in RECIPES:
            return "I don't have this recipe yet."

        added = []

        for item in RECIPES[recipe]["items"]:
            ref = find_item(item["name"], catalog)
            if not ref:
                continue

            for c in cart:
                if c["item_id"] == ref["id"]:
                    c["quantity"] += item["quantity"]
                    break
            else:
                cart.append({
                    "item_id": ref["id"],
                    "name": ref["name"],
                    "quantity": item["quantity"],
                    "unit_price": ref["price"]
                })

            added.append(ref["name"])

        session.userdata["cart"] = cart
        return f"Added ingredients for {recipe}: {', '.join(added)}."

    # ---------------------------------------------------
    # Tool: Place order
    # ---------------------------------------------------
    @function_tool
    async def finish_order(self):
        session = self.session
        cart = session.userdata.get("cart", [])

        if not cart:
            return "Your cart is empty."

        orders = load_orders()
        total = sum(c["unit_price"] * c["quantity"] for c in cart)
        order_id = next_order_id(orders)

        order = {
            "order_id": order_id,
            "created_at": time.time(),
            "status": "confirmed",
            "items": cart,
            "total": total,
        }

        orders.append(order)
        save_orders(orders)

        session.userdata["cart"] = []  # clear cart

        return f"Order {order_id} placed! Total ₹{total}. You can track it anytime."

    # ---------------------------------------------------
    # Tool: Track order
    # ---------------------------------------------------
    @function_tool
    async def track_order(self, order_id: Optional[str] = None):
        orders = load_orders()
        if not orders:
            return "No orders found."

        if order_id:
            order = next((o for o in orders if o["order_id"] == order_id), None)
            if not order:
                return "Order not found."
        else:
            order = orders[-1]

        diff = (time.time() - order["created_at"]) / 60

        if diff < 2:
            status = "confirmed"
        elif diff < 10:
            status = "preparing"
        elif diff < 20:
            status = "out for delivery"
        else:
            status = "delivered"

        order["status"] = status
        save_orders(orders)

        return f"Order {order['order_id']} is currently {status}."


# -----------------------------------------------------------
# Worker / Entrypoint (same as your Day-5 structure)
# -----------------------------------------------------------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    logger.info("Day 7 Grocery Agent Started.")

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice=VOICE_SDR,
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        userdata={"cart": []},
    )

    usage = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _metrics(ev: MetricsCollectedEvent):
        usage.collect(ev.metrics)

    agent = GroceryAgent()

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        )
    )

    await ctx.connect()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
