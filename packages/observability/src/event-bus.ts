/**
 * Typed event bus — the central observability hub.
 *
 * All simulation subsystems emit typed events here. Researchers subscribe
 * to specific event types, optionally filtered by population or projection ID.
 *
 * Zero overhead when no subscribers are attached (empty handler arrays
 * short-circuit before any computation).
 *
 * This is scientific instrumentation infrastructure, not application logging.
 * The distinction: a log records what happened; instrumentation enables
 * scientific measurement of phenomena.
 */

import type {
  EventBus,
  ObservabilityEvent,
  ObservabilityEventHandler,
  EventFilter,
  Unsubscribe,
} from '@snn/types';

interface Subscription {
  readonly handler: ObservabilityEventHandler;
  readonly filter?: EventFilter;
}

export class SpikeEventBus implements EventBus {
  private readonly subscribers = new Map<string, Subscription[]>();

  get subscriptionCount(): number {
    let count = 0;
    for (const subs of this.subscribers.values()) {
      count += subs.length;
    }
    return count;
  }

  subscribe<T extends ObservabilityEvent>(
    eventType: T['type'],
    handler: ObservabilityEventHandler<T>,
    filter?: EventFilter
  ): Unsubscribe {
    if (!this.subscribers.has(eventType)) {
      this.subscribers.set(eventType, []);
    }
    const sub: Subscription = { handler: handler as ObservabilityEventHandler, filter };
    this.subscribers.get(eventType)!.push(sub);

    return () => {
      const subs = this.subscribers.get(eventType);
      if (subs) {
        const idx = subs.indexOf(sub);
        if (idx !== -1) subs.splice(idx, 1);
      }
    };
  }

  emit(event: ObservabilityEvent): void {
    const subs = this.subscribers.get(event.type);
    if (!subs || subs.length === 0) return;

    for (const sub of subs) {
      if (this.matchesFilter(event, sub.filter)) {
        sub.handler(event);
      }
    }
  }

  clear(): void {
    this.subscribers.clear();
  }

  private matchesFilter(event: ObservabilityEvent, filter?: EventFilter): boolean {
    if (!filter) return true;

    if (filter.populationId !== undefined) {
      if ('populationId' in event && event.populationId !== filter.populationId) {
        return false;
      }
    }

    if (filter.projectionId !== undefined) {
      if ('projectionId' in event && event.projectionId !== filter.projectionId) {
        return false;
      }
    }

    return true;
  }
}
