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
    if not CATALOG_PATH.exists():
        return {"items": []}
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
    name = (name or "").lower().strip()
    for item in catalog.get("items", []):
        if name in item.get("name", "").lower():
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
- Set customer details (name, address, phone)

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
                try:
                    c["quantity"] = float(c["quantity"]) + float(quantity)
                except:
                    c["quantity"] = c.get("quantity", 0) + quantity
                break
        else:
            cart.append({
                "item_id": item["id"],
                "name": item["name"],
                "quantity": quantity,
                "unit_price": item.get("price", 0)
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
    # Tool: Show cart (detailed)
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
            try:
                line_total = float(c.get("unit_price", 0)) * float(c.get("quantity", 0))
            except:
                line_total = 0
            total += line_total
            # show integer quantity as int
            qty = int(c["quantity"]) if isinstance(c["quantity"], (int, float)) and float(c["quantity"]).is_integer() else c["quantity"]
            lines.append(f"{qty} x {c['name']} = ₹{line_total}")

        lines.append(f"Total = ₹{total}")
        return "\n".join(lines)

    # ---------------------------------------------------
    # Tool: What on the list (simple names)
    # ---------------------------------------------------
    @function_tool
    async def what_on_list(self):
        session = self.session
        cart = session.userdata.get("cart", [])
        if not cart:
            return "Your cart is empty."
        names = []
        for c in cart:
            qty = int(c["quantity"]) if isinstance(c["quantity"], (int, float)) and float(c["quantity"]).is_integer() else c["quantity"]
            names.append(f"{qty} x {c['name']}")
        return "Currently in your cart: " + ", ".join(names)

    # ---------------------------------------------------
    # Tool: List catalog or get item details
    # ---------------------------------------------------
    @function_tool
    async def list_catalog(self, query: Optional[str] = None):
        catalog = load_catalog()
        items = catalog.get("items", [])
        if not items:
            return "Catalog is currently empty."

        if query:
            ref = find_item(query, catalog)
            if not ref:
                return f"No items matching '{query}' found."
            # show basic details for the item
            return f"{ref.get('name','unknown')} — ₹{ref.get('price', 'N/A')} (id: {ref.get('id')})"
        # otherwise return a short list of item names
        names = [it.get("name", "unknown") for it in items]
        # limit output if very large
        if len(names) > 30:
            names = names[:30] + ["..."]
        return "Available: " + ", ".join(names)

    # ---------------------------------------------------
    # Tool: Recipe ingredients (improved matching + suggestions)
    # ---------------------------------------------------
    @function_tool
    async def add_recipe(self, recipe: str):
        session = self.session
        catalog = load_catalog()
        cart = session.userdata.get("cart", [])

        if not recipe or not recipe.strip():
            return "Which recipe would you like to add? Say something like 'peanut butter sandwich'."

        recipe_q = recipe.lower().strip()

        # 1) exact match
        if recipe_q in RECIPES:
            matched = recipe_q
        else:
            # 2) substring match (recipe name contains query or query contains recipe name)
            matched = None
            for rname in RECIPES.keys():
                if recipe_q in rname or rname in recipe_q:
                    matched = rname
                    break

        # 3) if still no match, provide suggestions
        if not matched:
            available = list(RECIPES.keys())
            suggestions = ", ".join(available[:6]) if available else "no recipes available"
            return f"I don't have that recipe. Available recipes: {suggestions}. Which one would you like?"

        # proceed to add ingredients for matched recipe
        added = []
        for item in RECIPES[matched]["items"]:
            ref = find_item(item["name"], catalog)
            if not ref:
                # continue silently if catalog missing the ingredient
                continue

            # find existing cart item
            for c in cart:
                if c["item_id"] == ref["id"]:
                    try:
                        c["quantity"] = float(c.get("quantity", 0)) + float(item.get("quantity", 1))
                    except:
                        c["quantity"] = item.get("quantity", 1)
                    break
            else:
                cart.append({
                    "item_id": ref["id"],
                    "name": ref["name"],
                    "quantity": item.get("quantity", 1),
                    "unit_price": ref.get("price", 0)
                })

            added.append(ref["name"])

        session.userdata["cart"] = cart
        if not added:
            return f"I tried adding ingredients for '{matched}', but none of the recipe items were found in the catalog."
        return f"Added ingredients for '{matched}': {', '.join(added)}."

    # ---------------------------------------------------
    # Tool: List available recipes
    # ---------------------------------------------------
    @function_tool
    async def list_recipes(self):
        if not RECIPES:
            return "No recipes available right now."
        names = list(RECIPES.keys())
        return "Available recipes: " + ", ".join(names)

    # ---------------------------------------------------
    # Tool: Set customer info (name, address, phone)
    # ---------------------------------------------------
    @function_tool
    async def set_customer_info(self, name: Optional[str] = None, address: Optional[str] = None, phone: Optional[str] = None):
        """
        Provide any of name, address, phone. Only the provided fields are updated.
        """
        session = self.session
        cust = session.userdata.get("customer", {"name": None, "address": None, "phone": None})

        if name:
            cust["name"] = name.strip()
        if address:
            cust["address"] = address.strip()
        if phone:
            cust["phone"] = phone.strip()

        session.userdata["customer"] = cust
        missing = [k for k, v in cust.items() if not v]
        if missing:
            return f"Saved details. Still missing: {', '.join(missing)}. Please provide them before placing order."
        return f"Saved customer details: {cust['name']}, {cust['address']}, {cust['phone']}."

    # ---------------------------------------------------
    # Tool: Place order (finish)
    # ---------------------------------------------------
    @function_tool
    async def finish_order(self):
        session = self.session
        cart = session.userdata.get("cart", [])

        if not cart:
            return "Your cart is empty."

        customer = session.userdata.get("customer", {"name": None, "address": None, "phone": None})
        missing = [k for k, v in customer.items() if not v]
        if missing:
            return f"Can't place order — please provide customer {', '.join(missing)} using set_customer_info."

        orders = load_orders()
        total = 0
        for c in cart:
            try:
                total += float(c.get("unit_price", 0)) * float(c.get("quantity", 0))
            except:
                pass
        order_id = next_order_id(orders)

        order = {
            "order_id": order_id,
            "created_at": time.time(),
            "status": "confirmed",
            "items": cart,
            "total": total,
            "customer": {
                "name": customer["name"],
                "address": customer["address"],
                "phone": customer["phone"],
            }
        }

        orders.append(order)
        save_orders(orders)

        session.userdata["cart"] = []  # clear cart
        # keep customer info so they don't have to re-enter next time
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

        cust = order.get("customer", {})
        cust_summary = ""
        if cust:
            cust_summary = f" for {cust.get('name','unknown')} to {cust.get('address','unknown')} (☎ {cust.get('phone','unknown')})"
        return f"Order {order['order_id']} is currently {status}{cust_summary}."


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
        userdata={"cart": [], "customer": {"name": None, "address": None, "phone": None}},
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
