# Payments Policy

## 1. Payment Overview

Customers may use the payment methods supported by the store during checkout.

Available payment methods depend on:

- Store configuration
- Customer location
- Order type
- Payment provider
- Product or seller restrictions
- Current payment availability

The AI assistant must use the payment information available from the configured payment or order system.

The AI must never invent payment status, transaction status, or payment confirmation.

---

## 2. Supported Payment Methods

Available payment methods may include:

- Credit cards
- Debit cards
- Digital payment methods
- Bank transfers
- Cash on delivery
- Other payment methods supported by the store

The actual payment options shown during checkout should be treated as the current source of truth.

---

## 3. Payment Status

### Payment Pending

The payment has been initiated but has not yet been confirmed as successful or failed.

The customer should avoid making repeated payments until the payment status is confirmed.

### Payment Successful

The payment provider or order system has confirmed that the payment was completed successfully.

### Payment Failed

The payment attempt was unsuccessful.

The customer may try an available payment method again after checking that no successful transaction was recorded.

### Payment Cancelled

The payment attempt was cancelled before completion.

### Payment Refunded

The payment amount has been refunded according to the refund process.

### Payment Partially Refunded

Only part of the original payment amount has been refunded.

---

## 4. Payment Failed

A payment may fail because of:

- Incorrect payment information
- Insufficient funds
- Bank restrictions
- Payment-provider issues
- Network problems
- Authentication failure
- Transaction limits
- Temporary technical problems

If a payment fails:

1. Check the payment status.
2. Confirm whether an order was created.
3. Avoid repeated payment attempts if the transaction status is unclear.
4. Try another supported payment method when appropriate.
5. Contact support if the issue continues.

The AI must not claim the exact reason for a failed payment unless the payment system provides that information.

---

## 5. Payment Pending

A pending payment means that the transaction has not received a final status.

Customers should:

1. Wait for the payment status to update.
2. Check whether an order was created.
3. Avoid making another payment for the same order while the status is unresolved.
4. Contact support if the payment remains pending beyond the expected processing period.

The AI must not treat a pending payment as successful or failed without confirmation.

---

## 6. Payment Successful but Order Not Created

If a customer reports that money was charged but no order was created:

1. Do not immediately make another payment.
2. Verify whether the payment was successful.
3. Check whether an order was created.
4. Check the payment transaction status.
5. Contact support if the order was not created.

The payment team may need to verify the transaction and determine whether the amount should be refunded or reconciled.

The AI must not promise a refund unless the refund process confirms eligibility.

---

## 7. Payment Failed but Amount Was Debited

Sometimes a payment may appear unsuccessful while the customer's account temporarily shows a debit.

In this situation:

1. Check the actual payment status.
2. Do not make repeated payments immediately.
3. Check whether the amount is automatically reversed.
4. Contact support if the amount is not reversed within the applicable processing period.

The exact reversal time depends on the payment provider and financial institution.

---

## 8. Duplicate Payment

If a customer believes they were charged more than once:

1. Verify the order ID.
2. Check the payment transaction history.
3. Identify whether multiple successful transactions exist.
4. Determine whether more than one order was created.
5. Escalate the issue when necessary.

The AI should not claim that a duplicate payment occurred without transaction information.

---

## 9. Duplicate Order

If a customer believes that the same order was created multiple times:

1. Verify the customer's order history.
2. Compare the order details.
3. Check associated payment transactions.
4. Determine whether multiple orders were actually created.
5. Follow the cancellation or refund process where applicable.

Do not cancel an order automatically without authorization and verification.

---

## 10. Payment Verification

Some payments may require additional verification by the payment provider or bank.

Verification can include:

- Bank authentication
- Payment-provider authentication
- Additional security checks

Customers should complete only the verification steps displayed by the official payment system.

Customer support should never request passwords, PINs, CVVs, or one-time authentication codes.

---

## 11. Card Payment

For card payments, customers should ensure that:

- The card is active.
- Sufficient funds or credit are available.
- Card details are entered correctly.
- The card supports the transaction type.
- Required authentication is completed through the official payment process.

If a card payment fails repeatedly, customers can try another supported payment method or contact their bank/payment provider.

---

## 12. Debit Card Payment

Debit-card payments may be declined because of:

- Insufficient account balance
- Transaction limits
- Bank restrictions
- Incorrect information
- Authentication failure
- Temporary banking issues

The customer should verify the transaction status before attempting another payment.

---

## 13. Credit Card Payment

Credit-card payments may be declined because of:

- Credit-limit restrictions
- Bank security checks
- Incorrect card information
- Merchant restrictions
- Authentication problems

Customers should contact their card issuer if the problem continues.

---

## 14. Digital Payment

Digital-payment availability depends on the payment providers configured by the store.

If a digital payment fails:

1. Check the transaction status.
2. Confirm whether an order was created.
3. Avoid making duplicate payments while the first transaction is unresolved.
4. Contact support if the amount was debited but the order was not created.

---

## 15. Bank Transfer

Bank-transfer payments may require additional processing time.

Customers should provide the applicable transaction reference when contacting support.

The support team can verify whether the payment has been received and associated with the correct order.

The AI should not confirm receipt of a bank transfer without system confirmation.

---

## 16. Cash on Delivery

Cash on delivery may be available for eligible orders and locations.

Availability can depend on:

- Delivery location
- Product
- Order value
- Seller
- Current store policy

The amount payable at delivery should be confirmed using the order information.

---

## 17. Payment Method Change

Customers may be able to change the payment method before completing checkout.

After an order is placed, changing the payment method may not be possible.

The support system should verify the current order status before providing instructions.

---

## 18. Payment Receipt

After a successful payment, a payment confirmation or receipt may be provided through the configured order or payment system.

If a customer cannot find a receipt:

1. Check the order confirmation.
2. Check the registered email or available account information.
3. Contact support if the receipt is unavailable.

The AI should only provide payment references confirmed by the system.

---

## 19. Transaction Reference

A payment transaction may have a unique transaction or payment reference.

When contacting support, the customer may provide the relevant transaction reference.

Only verified transaction references should be used for payment investigation.

---

## 20. Payment Reversal

A payment reversal may occur when a transaction cannot be completed and the payment provider returns the amount.

The timing of a reversal depends on the payment provider and financial institution.

Customers should check the transaction status before attempting another payment.

---

## 21. Refund After Payment Issue

If a payment was completed but the associated order could not be fulfilled, a refund may be applicable depending on the payment and order status.

The support system should:

1. Verify the payment.
2. Verify whether an order exists.
3. Determine refund eligibility.
4. Follow the configured refund process.

The AI must not promise a refund before eligibility is verified.

---

## 22. Payment Timeout

A payment session may expire if the transaction is not completed within the required period.

If a payment times out:

1. Check whether the payment was actually completed.
2. Check whether an order was created.
3. Verify the transaction status before attempting another payment.

---

## 23. Payment Provider Error

A payment provider may temporarily experience technical problems.

If a provider error occurs:

- Check the payment status.
- Avoid repeated transactions when the first transaction is unresolved.
- Try another supported payment method if appropriate.
- Contact support if money was debited or the problem continues.

The AI should not claim that a payment provider is experiencing an outage unless the configured system confirms it.

---

## 24. Payment Not Showing in Order

If a customer has completed payment but the order system does not show the payment:

1. Verify the order ID.
2. Verify the payment transaction reference.
3. Check the payment status.
4. Check whether the payment is still pending.
5. Escalate to the payment or support team when necessary.

Do not manually mark a payment as successful without authorization and system confirmation.

---

## 25. Incorrect Payment Amount

If the customer believes they were charged the wrong amount:

1. Verify the order total.
2. Check applicable discounts.
3. Check taxes and shipping charges.
4. Compare the payment transaction amount.
5. Escalate if the amounts do not match.

The AI should not calculate or promise an adjustment without verified order and payment data.

---

## 26. Payment Security

Customers should protect their payment and account information.

Never share the following with customer support or the chatbot:

- Passwords
- One-time passwords (OTP)
- PINs
- CVV/CVC
- Full card number when not required
- Banking passwords
- Authentication codes
- API keys
- Authentication tokens

Support agents should never request sensitive authentication credentials through normal support conversations.

---

## 27. Suspicious or Unauthorized Payment

If a customer reports an unauthorized or suspicious payment:

1. Treat the issue as a security-sensitive case.
2. Verify the available transaction information.
3. Ask the customer to contact the relevant bank or payment provider through its official channel when appropriate.
4. Escalate the case to human support.
5. Follow the account-security process.

Do not ask the customer to provide their password, OTP, PIN, CVV, or authentication codes.

---

## 28. Chargeback

A chargeback is a payment dispute initiated through the customer's card issuer or financial institution.

Chargeback cases may require specialist review.

If a customer asks about a chargeback:

- Collect only necessary order information.
- Do not request sensitive payment credentials.
- Escalate the case to the appropriate support or payment team.
- Provide only verified transaction information.

The AI should not make legal or financial determinations about a disputed transaction.

---

## 29. Payment Support Information

When handling a payment-related request, the support system may need:

- Customer ID
- Order ID
- Payment ID
- Transaction reference
- Payment status
- Payment amount
- Payment method
- Relevant timestamps

Only collect information necessary to investigate the issue.

---

## 30. AI Support Rules for Payments

The AI customer-support assistant must follow these rules:

1. Never invent payment status.
2. Never invent transaction IDs.
3. Never invent payment amounts.
4. Never claim a payment was successful without system confirmation.
5. Never claim a payment failed without system confirmation.
6. Never claim a refund was initiated without refund-system confirmation.
7. Never ask customers for passwords, OTPs, PINs, CVVs, or authentication credentials.
8. Do not expose sensitive payment information.
9. Do not manually mark transactions as successful.
10. Do not automatically retry an unresolved payment.
11. Escalate suspicious, unauthorized, disputed, or inconsistent payment cases.
12. Use verified payment and order information whenever available.
13. Clearly distinguish between pending, successful, failed, reversed, and refunded transactions.
14. When payment information is unavailable, clearly state that the transaction could not be verified.