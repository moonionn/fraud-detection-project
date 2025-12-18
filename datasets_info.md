# Dataset Overview

The dataset contains two main tables:

1. Transaction Table
2. Identity Table  
   They are joined by the key `TransactionID`.  
   Not every transaction has corresponding identity information.

## 1. Transaction Table

- **TransactionDT**: Time delta from a reference datetime (not an actual timestamp).
- **TransactionAMT**: Payment amount in USD.
- **ProductCD**: Product code for each transaction.
- **card1–card6**: Payment card information (type, category, issuer bank, country, etc.).
- **addr1, addr2**: Address information.
- **dist1, dist2**: Distance-related features (e.g., distance between billing and shipping addresses).
- **P_emaildomain / R_emaildomain**: Purchaser and recipient email domains.
- **C1–C14**: Counting features (e.g., number of addresses associated with a payment card). The exact meaning is masked.
- **D1–D15**: Time delta features (e.g., days since previous transaction).
- **M1–M9**: Matching indicators (e.g., whether card name matches billing name).
- **Vxxx**: Vesta-engineered features, including ranking, counting, and entity relations.

### Categorical Features – Transaction Table

ProductCD, card1–card6, addr1, addr2, P_emaildomain, R_emaildomain, M1–M9

## 2. Identity Table

Contains user identity and device-level information, including:

- **Network details**: IP address, ISP, proxy indicators.
- **Device fingerprints**: user agent, browser, OS, and version info.  
  These fields are anonymized for privacy reasons.

### Categorical Features – Identity Table

DeviceType, DeviceInfo, id_12–id_38

## Target Variable

**isFraud**: Binary label indicating whether a transaction is fraudulent (1 = fraud, 0 = legitimate).

## Files

- train_transaction.csv / train_identity.csv → training data
- test_transaction.csv / test_identity.csv → test data (predict isFraud)
