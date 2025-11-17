



interface Bill {
    /** BILL INFORMATION */

    id: string // e.g. "65820"
    number: string // e.g. "S.B. 213"

    state: string   // enum: e.g. "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"
    text_url: URL // e.g. "http://www.uconnruddcenter.org/resources/upload/docs/what/policy/legislation/AK_SB_213_State_Funding_For_School_Meals.pdf"
    summary: string // e.g. "Establishing the farm-to-school program in the Department of Natural Resources, and relating to school gardens, greenhouses, and farms"

    type: string // enum: e.g. "Proposed & Enacted Legislation"
    document_url: URL // e.g. "https://www.westlaw.com/Document/I95A46EE19B6111E187F99CB8BEA0B665/View/FullText.html?listSource=Search&list=PENDINGLEG-HISTORICAL&rank=10&sessionScopeId=556b4770216db273740c6eea6532747945726f34ee13b76512f5188037bcb672&originationContext=Search%20Result&transitionType=SearchItem&contextData=%28sc.Default%29&VR=3.0&RS=cblt1.0"
    filed_date: string // Date // e.g. "2012-01-01"

    /** primary bill sponsor */
    author: string // e.g. "A. Davis"
    cycle: number // e.g. "20122013"
    status: string // e.g. "enacted", "failed", "pending"
    status_date: string // Date // e.g. "2012-01-01"

    /** POLICY CLASSIFICATION  */

    /** Indicates inclusion as pro-prevention policy */
    pro: boolean
    outcome: "failed" | "passed"
    status_code:
    | "P_P" // Proposed Pro-obesity-prevention 
    | "F_P" // Failed Pro-obesity-prevention 
    | "P_A" // Proposed Anti-obesity-prevention 
    | "F_A" // Failed Anti-obesity-prevention
    | "P_N" // Proposed Neutral 
    | "F_N" // Failed Neutral

    category_environment: string // e.g. "S_E"
    topic: string   // e.g. enum: e.g. "SchoolOther"
}



/** LegiScan → Research Schema (per bill) */
interface BillRecord {
    /** BILL INFORMATION (direct from getBill unless noted) */

    id: string;                     // bill.bill_id (stringified)
    number: string | null;          // bill.number (e.g., "SB 213")
    state: string;                  // bill.state (e.g., "CA")

    text_url: string | null;        // newest bill.texts[].url (fallback bill.text_url)
    summary: string | null;         // bill.description (fallback bill.title)

    type: string | null;            // bill.bill_type (granular: e.g., "Bill", "Resolution", etc.)
    document_url: string | null;    // bill.url (LegiScan detail page)
    research_url?: string | null;   // bill.research_url (often state’s canonical page)
    filed_date: string | null;      // earliest bill.history[].date (ISO "YYYY-MM-DD")

    author: string | null;          // first sponsor name (bill.sponsors[0].name)
    sponsor_party?: string | null;  // bill.sponsors[0].party (if present)
    sponsors?: string[];            // bill.sponsors[].name (all sponsors)
    subjects?: string[];            // bill.subjects[].subject_name (topic hints)

    cycle: number | null;           // Number: session.year_start + year_end (e.g., 20212022)
    status: string;                 // mapped from bill.status → "introduced|engrossed|enrolled|passed|vetoed|failed|pending"
    status_date: string | null;     // bill.status_date (ISO)

    session_name?: string | null;        // bill.session.session_name
    session_year_start?: number | null;  // bill.session.year_start
    session_year_end?: number | null;    // bill.session.year_end
    last_action?: string | null;         // latest bill.history[].action
    last_action_date?: string | null;    // latest bill.history[].date (ISO)

    change_hash?: string | null;    // bill.change_hash (for versioning/dedupe)
    text_doc_ids?: number[];        // bill.texts[].doc_id (only if you plan to call getBillText later)

    /** POLICY CLASSIFICATION (manual or LLM-assisted later) */

    pro: boolean | null;            // inclusion flag: true if pro-prevention; null until coded
    outcome?: "failed" | "passed";  // derived from status (filled when final; omit while pending)
    status_code?:                   // combine outcome stage + intent (filled after coding)
    | "P_P" | "F_P"
    | "P_A" | "F_A"
    | "P_N" | "F_N";

    category_environment: string | null; // 5 envs (e.g., "S_E", "FB_E", "MS_E", "PA_E", "HC_WE")
    topic: string | null;                // 10 umbrella topics (e.g., "SchoolOther", "SchoolNutrition", ...)
}



//? do we want this necessarily?
// citation: string // e.g. "2012 Alabama House Bill No. 670, Alabama 2012 Regular Session" 
// leg_body: string // e.g. "S.B."

/**
Observations: 
- We will only record status_code, not each component as a boolean separately.
- Instead of "failed" and "passed", single "outcome" field with values "failed" or "passed".
- Instead of only "pro", use "neutrality" field with values "pro", "anti", or "neutral". The original "pro" doesn't consider if is "neutral", and might be falsy interpreted as "anti" if "pro" is false.

Questions: 
 - numstate and state are the same thing?


 */



// term mapping from reference document to new vocabulary. If the term is not found in the mapping, use the original term.
const mapping: Record<string, string> = {
    billnumber: "bill_number",
    statuscode: "status",
    billtitle: "bill_title",
    billtext: "bill_text",
    newdate: "new_date",
    recordid: "record_id",
    legbody: "leg_body",
    currentstatus: "current_status",
}







interface State2020To2025BillStatistics {
    state: string
    total_P_P: number
    total_F_P: number
    total_P_A: number
    total_F_A: number
    total_P_N: number
    total_F_N: number

    prerank_P_P: number
    rank_P_P: number
    prerank_F_P: number
    rank_F_P: number
    prerank_P_A: number
    rank_P_A: number
    prerank_F_A: number
    rank_F_A: number
    // these neutral bills were not ranked in the original study
    prerank_P_N: number
    rank_P_N: number
    prerank_F_N: number
    rank_F_N: number


    /** NEW stats */
    // state_n //? will we need this? what even is this?
    // instead of state_totalbills
    total_bills: number
    total_passed: number
    passrate: number
    // total_bills_enacted: number
    // total_bills_failed: number
    // total_bills_pending: number

    byYear: StateBillByYearStatistics[]
}


interface StateBillByYearStatistics {
    state: string
    year: number

    total_P_P: number
    total_F_P: number
    total_P_A: number
    total_F_A: number
    total_P_N: number
    total_F_N: number

    // n //? what is this?
    total_bills: number
    total_passed: number
    passrate: number

}

